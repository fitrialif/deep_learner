import os
import glob

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.engine.topology import InputLayer
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

train_dir = 'data/train'
val_dir = 'data/test'
nb_epoch = 50
batch_size = 64
img_width = 60
img_height = 160
fc_size = 1024
LAYERS_TO_FREEZE = 5


def get_nb_files(directory):
  """Get number of files by searching directory recursively"""
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt


def setup_to_transfer_learn(model, base_model):
	for layer in base_model.layers:
		layer.trainable = False
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


def setup_to_finetune(model):
	for layer in model.layers[:LAYERS_TO_FREEZE]:
		layer.trainable = False
	for layer in model.layers[LAYERS_TO_FREEZE:]:
		layer.trainable = True
	model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


def replace_last_layer(base_model, fc_size, nb_classes):
	"""Replace the last layer of a pre-trained ConvNet.
	Args:

	Returns:

	"""
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(fc_size, activation='relu')(x)
	predictions = Dense(nb_classes, activation='softmax')(x)
	model = Model(input=base_model.input, output=predictions)
	return model


def train():
	nb_train_samples = get_nb_files(train_dir)
	nb_classes = len(glob.glob(train_dir + "/*"))
	nb_val_samples = get_nb_files(val_dir)
	nb_epoch = 50
	batch_size = 32

	# Prepare data.
	train_datagen = ImageDataGenerator(
		rotation_range=30,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True
	)

	test_datagen = ImageDataGenerator(
		rotation_range=30,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True
	)

	train_generator = train_datagen.flow_from_directory(
		train_dir,
		target_size=(img_width, img_height),
		batch_size=batch_size,
	)

	validation_generator = test_datagen.flow_from_directory(
		val_dir,
		target_size=(img_width, img_height),
		batch_size=batch_size,
	)

	# setup model
	base_model = applications.VGG19(weights = "imagenet", include_top=False, input_shape=(img_width, img_height, 3))
	model = replace_last_layer(base_model, 256, nb_classes)

	# transfer learning
	setup_to_transfer_learn(model, base_model)

	history_tl = model.fit_generator(
		train_generator,
		nb_epoch=nb_epoch,
		samples_per_epoch=nb_train_samples,
		validation_data=validation_generator,
		nb_val_samples=nb_val_samples,
		class_weight='auto')

	# fine tuning
	setup_to_finetune(model)

	history_ft = model.fit_generator(
		train_generator,
		samples_per_epoch=nb_train_samples,
		nb_epoch=nb_epoch,
		validation_data=validation_generator,
		nb_val_samples=nb_val_samples,
		class_weight='auto')

	model.save('model.h5')


train()