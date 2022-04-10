import numpy as np
from utils import wav_stft, wav_istft, mag_to_patches, patches_to_mag
import soundfile as sf
import tensorflow as tf
from model import UNet
import time
import glob, os
from librosa.core import magphase

start_time = time.time()
patch_size = (513, 128)

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
data_path = '/media/samir/Secondary/Datasets/DSD100/Mixtures/Test/001 - ANiMAL - Clinic A/mixture.wav'
save_path = ''
model_path = '/media/samir/Secondary/Voice Separation/U-Net/models/models/UNet_Default_DSD10020220328-174223.h5'

# Loading Model
UNet = UNet()
model = UNet.get_model()
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

svs = model.predict(patches_input)
svs = np.squeeze(svs)

mag_svs = patches_to_mag(svs, (mag_size[0]-1, mag_size[1]))
fft = (mag_svs * mag_max) * phase[1:, :]

ifft = wav_istft(fft)

sf.write('test.wav', ifft, 8192, 'PCM_24')

print("Execution Time: %s s" % (time.time() - start_time))