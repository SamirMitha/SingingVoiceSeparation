from tensorflow.keras.layers import Input, LSTM, RNN, Dense, ReLU, Concatenate
from tensorflow.keras import Model
import tensorflow as tf
import numpy as np
from utils import TimeFreqMasking

class SVSRNN(Model):
	def __init__(self,
		img_size = (4, 512),
		num_rnn_layer = 3,
		num_hidden_units = [256, 256, 256]
		):

		# Initializations
		super(SVSRNN, self).__init__()
		self.img_size = img_size
		self.num_rnn_layer = num_rnn_layer
		self.num_hidden_units = num_hidden_units

	def get_model(self):
		return self.__forward()


	def srnn(self, input_tensor, img_size, num_hidden_units):
		lstm1 = LSTM(num_hidden_units[0],return_sequences=True,dropout=0.3,recurrent_dropout=0.1)(input_tensor)
		lstm2 = LSTM(num_hidden_units[1],return_sequences=True,dropout=0.3,recurrent_dropout=0.1)(lstm1)
		lstm3 = LSTM(num_hidden_units[2],return_sequences=False,dropout=0.3,recurrent_dropout=0.1)(lstm2)
		src1 = Dense(img_size[1], activation='relu')(lstm3)
		src2 = Dense(img_size[1], activation='relu')(lstm3)
		masked1 = TimeFreqMasking()([src1, src2, input_tensor])
		masked2 = TimeFreqMasking()([src2, src1, input_tensor])
		return([masked1, masked2])


	def __forward(self):
		inputs = Input(shape=self.img_size)
		outputs = self.srnn(inputs, self.img_size, self.num_hidden_units)
		model = Model(inputs, outputs)
		return(model)

if __name__ == '__main__':
	SVSRNN = SVSRNN()
	model = SVSRNN.get_model()
	print(model.summary())