import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers 
from keras.layers import BatchNormalization, Dense, Input
from keras.models import Model
from keras.utils.vis_utils import plot_model

K = 64
CP = K // 4
P = 8
allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])

if P < K:
    pilotCarriers = allCarriers[::K // P]  # Pilots is every (K/P)th carrier.
    dataCarriers = np.delete(allCarriers, pilotCarriers)

else:  # K = P
    pilotCarriers = allCarriers
    dataCarriers = []

mu = 2
payloadBits_per_OFDM = K * mu
SNRdb = 20
n_hidden_1 = 500  # 1st layer num features
n_hidden_2 = 250  # 2nd layer num features
n_hidden_3 = 120  # 3rd layer num features
n_output = 16  # every 16 bit are predicted by a model


def Modulation(bits):                                        
    bit_r = bits.reshape((int(len(bits)/mu), mu))                  
    return (2*bit_r[:,0]-1)+1j*(2*bit_r[:,1]-1)                                   

def OFDM_symbol(Data, pilot_flag):
    symbol = np.zeros(K, dtype=complex) # the overall K subcarriers 
    symbol[pilotCarriers] = pilotValue  # allocate the pilot subcarriers 
    symbol[dataCarriers] = Data  # allocate the pilot subcarriers
    return symbol

def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)

def addCP(OFDM_time):
    cp = OFDM_time[-CP:]               # take the last CP samples ...
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning

def channel(signal,channelResponse,SNRdb):
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdb/10)  
    noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
    return convolved + noise

def removeCP(signal):
    return signal[CP:(CP+K)]

def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)


def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest

def get_payload(equalized):
    return equalized[dataCarriers]


def PS(bits):
    return bits.reshape((-1,))

def ofdm_simulate(codeword, channelResponse,SNRdb):       
    OFDM_data = np.zeros(K, dtype=complex)
    OFDM_data[allCarriers] = pilotValue
    OFDM_time = IDFT(OFDM_data)
    OFDM_withCP = addCP(OFDM_time)
    OFDM_TX = OFDM_withCP
    OFDM_RX = channel(OFDM_TX, channelResponse,SNRdb)
    OFDM_RX_noCP = removeCP(OFDM_RX)

    # ----- target inputs ---
    symbol = np.zeros(K, dtype=complex)
    codeword_qam = Modulation(codeword)
    symbol[np.arange(K)] = codeword_qam
    OFDM_data_codeword = symbol
    OFDM_time_codeword = np.fft.ifft(OFDM_data_codeword)
    OFDM_withCP_cordword = addCP(OFDM_time_codeword)
    OFDM_RX_codeword = channel(OFDM_withCP_cordword, channelResponse,SNRdb)
    OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword)
    return np.concatenate((np.concatenate((np.real(OFDM_RX_noCP),np.imag(OFDM_RX_noCP))), np.concatenate((np.real(OFDM_RX_noCP_codeword),np.imag(OFDM_RX_noCP_codeword))))), abs(channelResponse) 


Pilot_file_name = 'Pilot_'+str(P)
if os.path.isfile(Pilot_file_name):
    print ('Load Training Pilots txt')
    # load file
    bits = np.loadtxt(Pilot_file_name, delimiter=',')
else:
    # write file
    bits = np.random.binomial(n=1, p=0.5, size=(K*mu, ))
    np.savetxt(Pilot_file_name, bits, delimiter=',')

pilotValue = Modulation(bits) 


channel_train = np.load('C:/Users/souad/Desktop/hopefully_my_last_dataset/train_h.response.npy', allow_pickle=True)
train_size = channel_train.shape[0]
channel_validation = np.load('C:/Users/souad/Desktop/hopefully_my_last_dataset/val_h.response.npy', allow_pickle=True)
validation_size = channel_validation.shape[0]
channel_test = np.load('C:/Users/souad/Desktop/hopefully_my_last_dataset/test_h.response.npy', allow_pickle=True)
test_size = channel_test.shape[0]

def training_gen(bs, SNRdb):
    while True:
        index = np.random.choice(np.arange(train_size), size=bs)
        H_total = channel_train[index] 
        input_samples = []
        input_labels = []
        for H in H_total:
            bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
            signal_output, para = ofdm_simulate(bits, H, SNRdb)
            input_labels.append(bits[0:16])
            input_samples.append(signal_output)
        yield (np.asarray(input_samples), np.asarray(input_labels))


def validation_gen(bs, SNRdb):
    while True:
        index = np.random.choice(np.arange(validation_size), size=bs)
        H_total = channel_validation[index]
        input_samples = []
        input_labels = []
        for H in H_total:
            bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
            signal_output, para = ofdm_simulate(bits, H, SNRdb)
            input_labels.append(bits[0:16])
            input_samples.append(signal_output)
        yield (np.asarray(input_samples), np.asarray(input_labels))

  
def testing_gen(bs, SNRdb):
    while True:
        index = np.random.choice(np.arange(test_size), size=bs)
        H_total = channel_test[index]
        input_samples = []
        input_labels = []
        for H in H_total:
            bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
            signal_output, para = ofdm_simulate(bits, H, SNRdb)
            input_labels.append(bits[0:16])
            input_samples.append(signal_output)
        yield (np.asarray(input_samples), np.asarray(input_labels)) 
        
        
def bit_err(y_true, y_pred):
    err = 1 - tf.reduce_mean(
        tf.reduce_mean(
            tf.cast( ##
                tf.equal(
                    tf.sign(
                        y_pred - 0.5),
                         ##
                    tf.cast(
                        tf.sign(
                            y_true - 0.5),
                        tf.float32)),
                tf.float32),
            1))
    return err


input_bits = Input(shape=(payloadBits_per_OFDM * 2,))
temp = BatchNormalization()(input_bits)
temp = Dense(n_hidden_1, activation='relu')(input_bits)
temp = BatchNormalization()(temp)
temp = Dense(n_hidden_2, activation='relu')(temp)
temp = BatchNormalization()(temp)
temp = Dense(n_hidden_3, activation='relu')(temp)
temp = BatchNormalization()(temp)
out_put = Dense(n_output, activation='sigmoid')(temp)
model = Model(input_bits, out_put)
model.compile(optimizer='adam', loss='mse', metrics=[bit_err])
model.summary()

model.fit(
    training_gen(1000,20),
    steps_per_epoch=50, 
    epochs=10000,
    validation_data=validation_gen(1000, 20),
    validation_steps=1,
    verbose=2)

model.save('Modelx')

#model1 = tf.keras.models.load_model('Modelx', custom_objects={'bit_err':bit_err}) # to load a model and evalute it


BER = []
for SNR in range(5, 31):
    y = model.evaluate(
        testing_gen(10000, SNR),
        steps=1
    )
    BER.append(y[1])
    print(y)
print(BER)