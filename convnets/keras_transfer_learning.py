import os
import glob

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.engine.topology import InputLayer
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, CSVLogger

from keras.applications.inception_v3 import preprocess_input

import matplotlib.pyplot as plt


from utils import ImageReadUtils, PlotUtils, TransferLearnUtils


train_dir = 'data/data-reid/train'
val_dir = 'data/data-reid/test'
nb_epoch = 10
batch_size = 128
# 2.32 only for inception
img_width = round(60*2.32)
img_height = round(160*2.32)
fc_size = 256
LAYERS_TO_FREEZE = 5
color_mode = 'grayscale'



def train():
	nb_train_samples = ImageReadUtils.getNbFiles(train_dir)
	nb_classes = len(glob.glob(train_dir + "/*"))
	nb_val_samples = ImageReadUtils.getNbFiles(val_dir)
	nb_epoch = 10
	batch_size = 64

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
	base_model = applications.VGG16(weights = "imagenet", 
		include_top=False, 
		input_shape=(img_width, img_height, 3))
	model = TransferLearnUtils.replaceClassificationLayer(base_model, 256, nb_classes)

	# transfer learning
	TransferLearnUtils.setupTransferLearn(base_model, model)

	early_stopping = EarlyStopping(monitor='val_loss',
		min_delta=0.01,
		patience=2,
		verbose=0,
		mode='auto')
	csv_logger_tl = CSVLogger('training_tl_inception.log')
	csv_logger_ft = CSVLogger('training_ft_inception.log')

	history_tl = model.fit_generator(
		train_generator,
		nb_epoch=25,
		samples_per_epoch=nb_train_samples,
		validation_data=validation_generator,
		nb_val_samples=nb_val_samples,
		class_weight='auto'])
#		callbacks=[early_stopping, csv_logger_tl])

	# history for accuracy
	plt.plot(history_tl.history['acc'])
	plt.plot(history_tl.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper_left')
	plt.show()
	# history for loss
	plt.plot(history_tl.history['loss'])
	plt.plot(history_tl.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper right')
	plt.show()

	# fine tuning
	TransferLearnUtils.setupFineTuning(model, LAYERS_TO_FREEZE)

	history_ft = model.fit_generator(
		train_generator,
		samples_per_epoch=nb_train_samples,
		nb_epoch=50,
		validation_data=validation_generator,
		nb_val_samples=nb_val_samples,
		class_weight='auto',
		callbacks=[early_stopping, csv_logger_ft])

	# history for accuracy
	plt.plot(history_ft.history['acc'])
	plt.plot(history_ft.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper_left')
	plt.show()
	# history for loss
	plt.plot(history_ft.history['loss'])
	plt.plot(history_ft.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper right')
	plt.show()

	model.save('vgg16_biomedical.h5')


train()