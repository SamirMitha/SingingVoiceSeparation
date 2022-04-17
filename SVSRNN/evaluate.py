import numpy as np
from utils import wav_stft, wav_istft, mag_to_patches, patches_to_mag
import soundfile as sf
import tensorflow as tf
from model import SVSRNN
import time
import glob, os
from librosa.core import magphase

start_time = time.time()
patch_size = (513, 4)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# Inputs
data_path = '/media/samir/Secondary/Datasets/DSD100/Mixtures/Dev/051 - AM Contra - Heart Peripheral/mixture.wav'
save_path = ''
model_path = '/media/samir/Secondary/Voice Separation/RNN/models/models/SVSRNN_Default_DSD100A20220416-233636.h5'

# Loading Model
SVSRNN = SVSRNN()
model = SVSRNN.get_model()
model.load_weights(model_path)

# Evaluation Loop
fft = wav_stft(data_path)
mag_size = fft.shape
mag, phase = magphase(fft)
mag_max = np.amax(mag)
mag = mag/mag_max

# Split to patches
patches = mag_to_patches(mag, patch_size)
patches_input = patches[:, 1:, :]

patches_input = patches_input.transpose(0,2,1)
print(patches_input.shape)
svs = model.predict(patches_input)
svs = np.squeeze(svs)
svs = svs.transpose(0,1,3,2)

mag_svs1 = patches_to_mag(svs[0,:,:,:], (mag_size[0]-1, mag_size[1]))
mag_svs2 = patches_to_mag(svs[1,:,:,:], (mag_size[0]-1, mag_size[1]))
fft1 = (mag_svs1 * mag_max) * phase[1:, :]
fft2 = (mag_svs2 * mag_max) * phase[1:, :]

ifft1 = wav_istft(fft1)
ifft2 = wav_istft(fft2)

sf.write('test1.wav', ifft1, 8192, 'PCM_24')
sf.write('test2.wav', ifft2, 8192, 'PCM_24')

print("Execution Time: %s s" % (time.time() - start_time))