import numpy as np
from librosa.core import stft, istft, magphase
import glob, os
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):

	def __init__(self,
		folder_path,
		batch_size=16,
		shuffle=True,
		patch_size=(512, 128)
		):

		# Initializations
		super(DataGenerator, self).__init__()
		self.folder_path = folder_path
		self.batch_size = batch_size
		self.list_IDs = os.listdir(self.folder_path)
		self.patch_size = patch_size
		self.shuffle = shuffle
		self.on_epoch_end()

	def __len__(self):
		return int(np.floor(len(self.list_IDs) / self.batch_size))


	def __getitem__(self, index):

		# Generates indexes of the batched data
		indexes = self.indexes[index * self.batch_size:(index+1) * self.batch_size]

		# Get list of IDs
		list_IDs_temp = [self.list_IDs[k] for k in indexes]

		# Generate data
		X, y = self.__data_generation(list_IDs_temp)
		return X, y


	def on_epoch_end(self):

		self.indexes = np.arange(len(self.list_IDs))

		if(self.shuffle):
			np.random.shuffle(self.indexes)


	def __data_generation(self, list_IDs_temp):
		# Initialization
		X = np.empty((self.batch_size, *self.patch_size))
		y = np.empty((self.batch_size, *self.patch_size))

		# Generate data
		for i, ID in enumerate(list_IDs_temp):

			mix = np.load(self.folder_path + ID)
			vocals = np.load(self.folder_path.replace('mix/', 'vocals/') + ID)

			# Store sample
			X[i, ] = mix

			# Store class
			y[i, ] = vocals

			# output vector
		return X, y
