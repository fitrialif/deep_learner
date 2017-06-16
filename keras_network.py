from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras import backend as K
import matplotlib.pyplot as plt

img_width, img_height = 116, 116

train_data_dir = 'cache_dir/train'
validation_data_dir = 'cache_dir/test'
nb_train_samples = 20000
nb_validation_samples = 8000
epochs = 50
batch_size = 128

if K.image_data_format() == 'channels_first':
	input_shape = (1, img_width, img_height)
else:
	input_shape = (img_width, img_height, 1)


def create_model(learning=0.2):
	model = Sequential()
	model.add(Conv2D(64, (5, 5), input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(128, (5, 5)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	op = optimizers.SGD(lr=learning)

	model.compile(loss='binary_crossentropy', optimizer=op, metrics=['accuracy'])
	return model

#batch_size = 16

# augmentation configuration for training
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# augmentation configuration for testing
test_datagen = ImageDataGenerator(rescale=1.255)

epochs = [10, 50]
batch_size = [128, 256]
learning_rates = [0.01, 0.1]

# list comprehension to get all params values. TODO: parametrize for an arbitrary number of parameters
list_params = [(p1, p2, p3) for p1 in epochs for p2 in batch_size for p3 in learning_rates]
# todo: parametrize this list
params = ['epochs', 'batch_size', 'learning_rates']

# create a list of dict
list_dict = []
for el in list_params:
	list_dict.append(dict(zip(params, el)))

# for each dict in the list, call the train_generator, validation_generator, with the keys in that particular dict
for d in list_dict:
	batch_size = d.get('batch_size')
	epochs = d.get('epochs')
	learning_rates = d.get('learning_rates')
	# Create model.
	model = create_model(learning_rates)
	train_generator = train_datagen.flow_from_directory('cache_dir/train', target_size=(img_width, img_height), color_mode='grayscale', batch_size=batch_size, class_mode='binary')
	validation_generator = test_datagen.flow_from_directory('cache_dir/test', target_size=(img_width, img_height), color_mode='grayscale', batch_size=batch_size, class_mode='binary')
	history = model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=epochs, validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size)
	# history for accuracy
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper_left')
	plt.show()
	# history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper right')
	plt.show()