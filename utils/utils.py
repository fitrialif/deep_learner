import os
import glob
import itertools
import numpy as np
import matplotlib.pyplot as plt

from keras import optimizers
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

#from sklearn.metrics import confusion_matrics


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
"""
	def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues)
		""""""
		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(classes))
		plt.xticks(tick_marks, classes, rotation=45)
		plt.yticks(tick_marks, classes)

		if normalize:
			cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
			print("Normalized confusion matrix")
		else:
			print("Confusion matrix, without normalization")

		print(cm)

		thresh = cm.max() / 2.
	    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
	        plt.text(j, i, cm[i, j],
	                 horizontalalignment="center",
	                 color="white" if cm[i, j] > thresh else "black")

	    plt.tight_layout()
	    plt.ylabel('True label')
	    plt.xlabel('Predicted label')"""


class TransferLearnUtils:
	def setupTransferLearn(base_model, model, optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']):
		""""""
		for layer in base_model.layers:
			layer.trainable = False
		
		model.compile(optimizer=optimizer,
			loss=loss,
			metrics=metrics)
#		return model

	def setupFineTuning(model, layers_to_freeze=0, optimizer=optimizers.SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy']):
		""""""
		for layer in model.layers[:layers_to_freeze]:
			layer.trainable = False
		for layer in model.layers[layers_to_freeze:]:
			layer.trainable = True
		model.compile(optimizer=optimizer,
			loss=loss,
			metrics=metrics)
#		return model

	def replaceClassificationLayer(base_model, fc_size, nb_classes):
		x = base_model.output
		x = GlobalAveragePooling2D()(x)
		x = Dense(fc_size, activation='relu')(x)
		predictions = Dense(nb_classes, activation='softmax')(x)
		model = Model(input=base_model.input, output=predictions)

		return model