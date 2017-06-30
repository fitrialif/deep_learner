from keras.models import Sequential

from keras.layers.advanced_activations import LeakyReLU

from keras import backend as K


class GenerativeAdversarialNetwork:

	""" """
	def __init__(self, image_width, image_height, image_channels, discriminative_depth, generativeDepth, dropout):
		self.img_width = image_width
		self.img_height = image_height
		self.img_channels = image_channels
		self.dDepth = discriminativeDepth
		self.dropout = dropout


	def setInputShape(self):
		if K.image_data_format() == 'channels_first':
			self.input_shape = (self.img_channels, self.img_width, self.img_height)
		else:
			self.input_shape = (self.img_width, self.img_height, self.img_channels)


	def createDiscriminativeModel(self):
		self.discriminativeModel = Sequential()
		self.discriminativeModel.add(Conv2D(self.depth*1, 5, strides=2, input_shape=self.input_shape, padding='same', activation=LeakyReLU(alpha=0.2)))
		self.discriminativeModel.add(Dropout(self.dropout))
		self.discriminativeModel.add(Conv2D(self.depth*2, 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
		self.discriminativeModel.add(Dropout(self.dropout))
		self.discriminativeModel,add(Conv2D(self.depth*4, 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
		self.discriminativeModel.add(Dropout(self.dropout))
		self.discriminativeModel.add(Conv2D(self.depth*8, 5, strides=1, padding='same', activation=LeakyReLU(alpha=0.2)))
		self.discriminativeModel.add(Dropout(self.dropout))
		self.discriminativeModel.add(Flatten())
		self.discriminativeModel.add(Dense(1))+
		self.discriminativeModel.add(Activation('sigmoid'))


	def createGenerativeModel(self):
		self.generativeModel = Sequential()
		self.