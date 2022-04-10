from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Concatenate, Conv2DTranspose, Dropout, ReLU
from tensorflow.keras import Model

class UNet(Model):
	def __init__(self,
		img_size = (512, 128),
		num_filters = (1, 16, 32, 64, 128, 256, 512),
		kernel_size = 5,
		strides = 2
		):

		# Initializations
		super(UNet, self).__init__()
		self.img_size = img_size
		self.num_filters = num_filters
		self.kernel_size = kernel_size
		self.strides = strides

	def get_model(self):
		return self.__forward()


	def conv_block(self, input_tensor, num_filters, kernel_size, strides):
		conv1 = Conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding='same')(input_tensor)
		bn1 = BatchNormalization()(conv1)
		lrelu1 = LeakyReLU(alpha=0.2)(bn1)
		return (lrelu1)


	def deconv_block(self, input_tensor, num_filters, kernel_size, strides, batch_norm = True, dropout = True):
		deconv1 = Conv2DTranspose(filters=num_filters, kernel_size=kernel_size, strides=strides, padding='same')(input_tensor)
		if batch_norm == True:
			bn1 = BatchNormalization()(deconv1)
		else:
			bn1 = deconv1
		if dropout == True:
			drop1 = Dropout(0.5)(bn1)
		else:
			drop1 = bn1
		relu1 = ReLU()(drop1)
		return(relu1)

	def u_net(self, input_tensor, num_filters, kernel_size, strides):
		conv1 = self.conv_block(input_tensor, num_filters[1], kernel_size, strides)
		conv2 = self.conv_block(conv1, num_filters[2], kernel_size, strides)
		conv3 = self.conv_block(conv2, num_filters[3], kernel_size, strides)
		conv4 = self.conv_block(conv3, num_filters[4], kernel_size, strides)
		conv5 = self.conv_block(conv4, num_filters[5], kernel_size, strides)
		conv6 = self.conv_block(conv5, num_filters[6], kernel_size, strides)
		deconv1 = self.deconv_block(conv6, num_filters[5], kernel_size, strides)
		cat1 = Concatenate(axis=3)([deconv1, conv5])
		deconv2 = self.deconv_block(cat1, num_filters[4], kernel_size, strides)
		cat2 = Concatenate(axis=3)([deconv2, conv4])
		deconv3 = self.deconv_block(cat2, num_filters[3], kernel_size, strides)
		cat3 = Concatenate(axis=3)([deconv3, conv3])
		deconv4 = self.deconv_block(cat3, num_filters[2], kernel_size, strides, True, False)
		cat4 = Concatenate(axis=3)([deconv4, conv2])
		deconv5 = self.deconv_block(cat4, num_filters[1], kernel_size, strides, True, False)
		cat5 = Concatenate(axis=3)([deconv5, conv1])
		deconv6 = self.deconv_block(cat5, num_filters[0], kernel_size, strides, False, True)
		return(deconv6)


	def __forward(self):
		inputs = Input(shape=(512, 128, 1))
		output_mask = self.u_net(inputs, self.num_filters, self.kernel_size, self.strides)
		output = output_mask * inputs
		model = Model(inputs, output)
		return(model)

if __name__ == '__main__':
	UNet = UNet()
	model = UNet.get_model()
	print(model.summary())