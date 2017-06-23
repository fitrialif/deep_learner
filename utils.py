import os
import glob
import matplotlib.pyplot as plt


from keras import optimizers
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model


class ImageReadUtils:
	def getNbFiles(directory):
		""" Get the number of files by searching folder recursively."""
		if not os.path.exists(directory):
			return 0 # sostituire con eccezione
		fileCount = 0
		for r, dirs, files in os.walk(directory):
			for dr in dirs:
				fileCount += len(glob.glob(os.path.join(r, dr + "/*")))
		return fileCount

	def getNbClasses(trainDir):
		return len(glob.glob(trainDir + "/*"))


class PlotUtils:
	def plotHistoryResults(history):
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


class TransferLearnUtils:
	def setupTransferLearn(base_model, model, optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']):
		""""""
		for layer in base_model.layers:
			layer.trainable = False
		
		model.compile(optimizer=optimizer,
			loss=loss,
			metrics=metrics)

	def setupFineTuning(model, layers_to_freeze=0, optimizer=optimizers.SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy']):
		""""""
		for layer in model.layers[:layers_to_freeze]:
			layer.trainable = False
		for layer in model.layers[layers_to_freeze:]:
			layer.trainable = True
		model.compile(optimizer=optimizer,
			loss=loss,
			metrics=metrics)

	def replaceClassificationLayer(base_model, fc_size, nb_classes):
		x = base_model.output
		x = GlobalAveragePooling2D()(x)
		x = Dense(fc_size, activation='relu')(x)
		predictions = Dense(nb_classes, activation='softmax')(x)
		model = Model(input=base_model.input, output=predictions)

		return model