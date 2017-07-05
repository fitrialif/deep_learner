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
	def arrange_dataset(directory, file_extension='*.jpg'):
		# Directory should be the dataset directory.
		if not os.path.exists(directory):
			return 0
		cur_dir = os.getcwd()
		os.chdir(directory)
		work_dir = os.getcwd()
		file_list = glob.glob('*.jpg')
		train_dir = os.path.abspath('train')
		test_dir = os.path.abspath('test')

		if not os.path.exists(train_dir):
			os.mkdir(train_dir)
		if not os.path.exists(test_dir):
			os.mkdir(test_dir)

		os.chdir(cur_dir)

		# set init label
		init_label = file_list[0][0:4]
		current_list = []
		for file in file_list:
			# TODO: param label with regex and boundaries
			label = file[0:4]
			# if label is the same, append to list, else evolve list
			if label == init_label:
				if not os.path.exists(os.path.join(train_dir, label)):
					os.mkdir(os.path.join(train_dir, label))
				if not os.path.exists(os.path.join(test_dir, label)):
					os.mkdir(os.path.join(test_dir, label))
				current_list.append(file)
			if (label != init_label) or (file == file_list[-1]):
				partial_train, partial_test = ImageReadUtils.arrange_dataset_2(current_list)
#				return partial_train, partial_test
				for train_file in partial_train:
					os.rename(os.path.join(work_dir, train_file), os.path.join(train_dir, init_label, train_file))
				for test_file in partial_test:
					os.rename(os.path.join(work_dir, test_file), os.path.join(test_dir, init_label, test_file))
				current_list[:] = []
				init_label = label
				current_list.append(file)



	def arrange_dataset_2(file_list, train_split=0.8):
		# TODO: CHECK IF FILE LIST IS PROPERLY LONG (at least 5 identities). if it is not, train_split = round(leng)
		if len(file_list) == 2:
			train_split = 0.5
		elif len(file_list) == 3:
			train_split = 0.66
		elif len(file_list) == 4:
			train_split = 0.75
		random_set = np.random.permutation(len(file_list))
#		train_size = round(len(file_list)*train_split)
#		test_size = len(file_list) - train_size
		train_list = random_set[:round(len(random_set)*0.8)]
		test_list = random_set[-(len(file_list) - len(train_list))::]
		train_images = []
		test_images = []
		for index in train_list:
			train_images.append(file_list[index])
		for index in test_list:
			test_images.append(file_list[index])
		return train_images, test_images


	def getNbFiles(directory):
		""" Get the number of files by searching folder recursively."""
		if not os.path.exists(directory):
			return 0
		fileCount = 0
		for r, dirs, files in os.walk(directory):
			for dr in dirs:
				fileCount += len(glob.glob(os.path.join(r, dr + "/*")))
		return fileCount

	def getNbClasses(trainDir):
		return len(glob.glob(trainDir + "/*"))

		## todo: chiamata ricorsiva
	def split_image_directory(directory, img_extension, test_size):
		image_list = glob.glob(os.path.join(directory, img_extension))
		train_samples, validation_samples = train_test_split(image_list, test_size=test_size)
		return train_samples, validation_samples

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
	def setupTransferLearn(base_model, model, optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy']):
		""""""
		for layer in base_model.layers:
			layer.trainable = False
		
		model.compile(optimizer=optimizer,
			loss=loss,
			metrics=metrics)
#		return model

	def setupFineTuning(model, layers_to_freeze=0, optimizer=optimizers.SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy']):
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