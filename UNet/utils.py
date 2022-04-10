import os
import librosa
import soundfile as sf
from librosa.core import stft, istft, magphase
from time import time
import numpy as np
from math import ceil
  
# SAMPLE_RATE=8192
# WINDOW_SIZE=1024
# HOP_LENGTH=768

# PATCH_SIZE=128
# EPOCH = 20 # test
# BATCH = 16
# SAMPLE_STRIDE = 10

# need vocals wav
# need instrumental wav
# need full song wav
# need vocals fft
# need full song fft
# need to normalize ffts from full song max
# need downsampled wavs
# create proprocessed dataset folder

def wav_stft(input, sampling_rate=8192):
	wav, _ = librosa.load(input, sr=sampling_rate)
	fft = stft(wav, n_fft=1024, hop_length=768)
	return(fft)


def wav_istft(input, sampling_rate=8192):
	fft = input
	ifft = istft(fft, n_fft=1024, hop_length=768)
	return(ifft)


def preprocess(DSD100_path, output_folder):
	paths = os.listdir(DSD100_path + 'Mixtures/Dev')
	for path in paths:
		print(path)
		mix = wav_stft(DSD100_path + 'Mixtures/Dev/' + path + '/mixture.wav')
		vocals = wav_stft(DSD100_path + 'Sources/Dev/' + path + '/vocals.wav')
		mix_mag, _ = magphase(mix)
		vocals_mag, _ = magphase(vocals)
		mix_mag_norm = mix_mag/np.amax(mix_mag)
		vocals_mag_norm = vocals_mag/np.amax(mix_mag)
		os.mkdir(output_folder + path)
		np.save(output_folder + path + '/mix.npy', mix_mag_norm)
		np.save(output_folder + path + '/vocals.npy', vocals_mag_norm)


def mag_to_patches(mag, patch_size):
	mag_size = mag.shape
	m = int(ceil(mag_size[0]/patch_size[0]))
	n = int(ceil(mag_size[1]/patch_size[1]))

	mag_new_size = (patch_size[0] * m, patch_size[1] * n)

	padding = tuple(np.subtract(mag_new_size, mag_size))
	padding_full = ((0, padding[0]), (0, padding[1]))

	padded = np.pad(mag, pad_width = padding_full, mode='reflect')

	num_patches = m * n

	patches = np.zeros((num_patches, *patch_size))

	for i in range(m):
		for j in range(n):
			bbox1 = i * patch_size[0]
			bbox2 = (i+1) * patch_size[0]
			bbox3 = j * patch_size[1]
			bbox4 = (j+1) * patch_size[1]
			patch = padded[bbox1:bbox2, bbox3:bbox4]
			patches[n * i + j, :, :] = patch

	return (patches)


def patches_to_mag(patches, mag_size):
	# converts patches to image to be used during evaluation
	patch_size = patches.shape
	patch_size = (patch_size[1], patch_size[2])

	m = int(ceil(mag_size[0]/patch_size[0]))
	n = int(ceil(mag_size[1]/patch_size[1]))
	padded = np.zeros((m*patch_size[0], n*patch_size[1]))

	for i in range(m):
		for j in range(n):
			bbox1 = i * patch_size[0]
			bbox2 = (i+1) * patch_size[0]
			bbox3 = j * patch_size[1]
			bbox4 = (j+1) * patch_size[1]
			patch = patches[n * i + j, :, :]
			padded[bbox1:bbox2, bbox3:bbox4] = patch

	mag = padded[0:(mag_size[0]), 0:(mag_size[1])]

	return (mag)

def preprocess_to_patches(preprocess_folder, output_folder):
	paths = os.listdir(preprocess_folder)
	for path in paths:
		if(not os.path.isdir(output_folder) or not os.listdir(output_folder)):
			os.mkdir(output_folder)
			os.mkdir(output_folder + 'mix')
			os.mkdir(output_folder + 'vocals')
		print(path)
		mix = preprocess_folder + path + '/mix.npy'
		vocals = preprocess_folder + path + '/vocals.npy'

		mix_fft = np.load(mix)
		vocals_fft = np.load(vocals)

		mix_mag, _ = magphase(mix_fft)
		vocals_mag, _ = magphase(vocals_fft)

		mix_mag_max = np.amax(mix_mag)
		mix_mag = mix_mag/mix_mag_max
		vocals_mag = vocals_mag/mix_mag_max
		
		mix_patches = mag_to_patches(mix_mag, (512, 128))
		vocals_patches = mag_to_patches(vocals_mag, (512, 128))

		for i in range(mix_patches.shape[0]):
			np.save(output_folder + 'mix/' + path + '_sample_' + str(i) + '.npy', mix_patches[i, :, :])
			np.save(output_folder + 'vocals/' + path + '_sample_' + str(i) + '.npy', vocals_patches[i, :, :])


#preprocess_to_patches('/media/samir/Secondary/Voice Separation/U-Net/DSD100_preprocessed/', 'training/')

# preprocess('/media/samir/Secondary/Datasets/DSD100/', 'DSD100_preprocessed/')
# a = DSD100_stft('test/vocals.wav')
# np.save('test.npy', a)
# b = DSD100_istft('test.npy')
# sf.write('test.wav', b, 8192, 'PCM_24')