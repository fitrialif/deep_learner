import os, glob, sys
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
#from keras import backend as K
from sklearn.model_selection import train_test_split
from keras import optimizers
from pandas import DataFrame
from pandas import concat

def set_image_format(img_rows, img_cols, img_channels, keras_backend):
    """
    """
    if keras_backend == 'channels_first':
        input_shape = (img_channels, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, img_channels)


def get_nb_files(root_folder):
    """
    Get the number of files in a folder's hierarchy.
    Arguments:
        root_folder: A string which represents the root folder.
    Returns:
        file_count: An integer which represents the number of files in the folder's hierarchy.
    """
    if not os.path.exists(root_folder):
        sys.exit('Root folder does not exist.')
    file_count = 0
    for r, dirs, files in os.walk(root_folder):
        for dr in dirs:
            file_count += len(glob.glob(os.path.join(r, dr + "/*")))
    return file_count


def get_labels(train_folder, validation_folder):
    """Get the number of labels.
    Arguments:
        train_folder: A string which represents the relative path of the train folder.
        validation_folder: A string which represents the relative path of the validation folder.
    Return:
        num_labels: An integer which represents the number of labels for the images on which the network will be trained.
    """
    if not os.path.exists(train_folder):
        sys.exit('Train folder does not exist.')
    if not os.path.exists(validation_folder):
        sys.exit('Validation folder does not exist.')
    if not (len(glob.glob(train_folder + "/*")) == len(glob.glob(validation_folder + "/*"))):
        sys.exit('The number of training labels is different from the number of validation labels.')
    else:
        num_labels = len(glob.glob(train_folder + "/*"))
        return num_labels





def makedirs_wrapper(folder):
    """Simple wrapper around the os.makedirs function. It automatically checks if the argument folder exists, and creates it if not.
    """
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError:
            pass


def split_each_label(root_folder, min_examples=0):
    """
    Algorithm:
    1. Check if root_folder exists
    2. Create train_folder and test_folder
    3. TODO: CREATE THE SUBFOLDER STRUCTURE FROM STRING NAME
    TODO: BALANCE DATABASE
    4. For each subfolder in the subfolder structure, check length and create sub-train and sub-test folders
    5. Split training and testing set
    6. Move images into proper folders
    7. TODO: remove unused folders
    """
    if not os.path.exists(root_folder):
        sys.exit("Root folder does not exist.")
    # Check if train and test folder exist. If not, create them.
    train_folder = os.path.abspath(os.path.join(root_folder, 'train'))
    test_folder = os.path.abspath(os.path.join(root_folder, 'test'))
    makedirs_wrapper(train_folder)
    makedirs_wrapper(test_folder)
    for root, dirs, files in os.walk(root_folder):
        for d in dirs:
            act_len = len(glob.glob(os.path.join(root, d + "/*")))
            dir_path = os.path.join(root_folder, d)
            # Ugly hack
            if d == 'train' or d == 'test':
                break
            else:
                d_train_folder = os.path.join(train_folder, d)
                d_test_folder = os.path.join(test_folder, d)
                makedirs_wrapper(d_train_folder)
                makedirs_wrapper(d_test_folder)
            if act_len > min_examples:
                x = y = os.listdir(dir_path)
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
                if x_test:
                    for x in x_train:
                        os.rename(os.path.join(dir_path, x), os.path.join(d_train_folder, x))
                    for x in x_test:
                        os.rename(os.path.join(dir_path, x), os.path.join(d_test_folder, x))




# Courtesy of Jason Brownlee
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def plot_training(history):
    val_acc = history.history['val_acc']
    val_loss = history.history['val_loss']
    epochs = range(len(history.history['acc']))
    plt.plot(epochs, val_acc, 'r')
    plt.title('Validation accuracy')
    plt.figure()
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Validation loss')
    plt.show()