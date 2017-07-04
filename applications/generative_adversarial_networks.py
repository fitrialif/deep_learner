from keras.models import Sequential

from keras.layers.advanced_activations import LeakyReLU

from keras import backend as K


class GenerativeAdversarialNetwork:

	""" """
	def __init__(self, image_width, image_height, image_channels):
		self.img_width = image_width
		self.img_height = image_height
		self.img_channels = image_channels


	def setInputShape(self):
		if K.image_data_format() == 'channels_first':
			self.input_shape = (self.img_channels, self.img_width, self.img_height)
		else:
			self.input_shape = (self.img_width, self.img_height, self.img_channels)


	def createDiscriminativeModel(self, depth, dropout, loss, optimizer, metrics):
		self.discriminativeModel = Sequential()
		self.discriminativeModel.add(Conv2D(depth*1, 5, strides=2, input_shape=self.input_shape, padding='same', activation=LeakyReLU(alpha=0.2)))
		self.discriminativeModel.add(Dropout(dropout))
		self.discriminativeModel.add(Conv2D(depth*2, 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
		self.discriminativeModel.add(Dropout(dropout))
		self.discriminativeModel,add(Conv2D(depth*4, 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
		self.discriminativeModel.add(Dropout(dropout))
		self.discriminativeModel.add(Conv2D(depth*8, 5, strides=1, padding='same', activation=LeakyReLU(alpha=0.2)))
		self.discriminativeModel.add(Dropout(dropout))
		self.discriminativeModel.add(Flatten())
		self.discriminativeModel.add(Dense(1))
		self.discriminativeModel.add(Activation('sigmoid'))


	def createGenerativeModel(self, depth, dropout):
		self.generativeModel = Sequential()
		self.generativeModel.add(Dense(self.img_width*self.img_height*depth))
		self.generativeModel.add(BatchNormalization(momentum=0.9))
		self.generativeModel.add(Activation('relu'))
		self.generativeModel.add(Reshape(self.img_width, self.img_height, depth))
		self.generativeModel.add(Dropout(dropout))
		self.generativeModel.add(UpSampling2D())
		self.generativeModel.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
		self.generativeModel.add(BatchNormalization(momentum=0.9))
		self.generativeModel.add(Activation('relu'))
		self.generativeModel.add(UpSampling2D())
		self.generativeModel.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
		self.generativeModel.add(BatchNormalization(momentum=0.9))
		self.generativeModel.add(Activation('relu'))
		self.generativeModel.add(Conv2DTranspose(1, 5, padding='same'))
		self.generativeModel.add(Activation('sigmoid'))
		self.generativeModel.summary()


	def createAdversarialModel(self, loss, optimizer, metrics):
		self.adversarialModel = Sequential()