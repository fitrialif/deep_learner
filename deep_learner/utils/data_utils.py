import os

from keras import backend as K
from keras.datasets import *
from keras.utils.data_utils import get_file


__dataset_dictionary = {
    'cifar10': cifar10.load_data(),
    'cifar100': cifar100.load_data(),
    'imdb': imdb.load_data(),
    'reuters': reuters.load_data(),
    'mnist': mnist.load_data(),
    'fashion': fashion_mnist.load_data(),
    'boston': boston_housing.load_data()
}


def get_nb_files(root_folder):
    """Gets the number of files in a folder's hierarchy.

    # Arguments:
        root_folder (string): The relative path of the root folder.
    # Returns:
        file_count (integer): The number of files in the folder's hierarchy.
    """
    if not os.path.exists(root_folder):
        raise ValueError('Could not find the specified %s folder.' % (root_folder))

    file_count = 0
    for r, dirs, files in os.walk(root_folder):
        for dr in dirs:
            file_count += len(glob.glob(os.path.join(r, dr + "/*")))
    return file_count


def load_keras_dataset(dataset='cifar10'):
    """Loads a Keras dataset. Wrapper around dataset-specific methods.

    For the specific type of the return, check Keras documentation.

    # Arguments:
        dataset (string): The name of the dataset.

    # Returns:
        x_train: An object which contains the training data.
        y_train: An object which contains the training data labels.
        x_test: An object which contains the validation data.
        x_test: An object which contains the validation data labels.
    """
    if dataset not in __dataset_dictionary:
        raise ValueError('Could not find a valid dataset for argument %s.' % (dataset))
    
    return (x_train, y_train), (x_test, y_test) = __dataset_dictionary[dataset]


def load_external_dataset(dataset_name, dataset_origin, num_train_samples, num_batches, img_channels=3, img_width=32, img_height=32):
    """Loads a generic dataset.

    # Arguments:

    # Returns:

    """