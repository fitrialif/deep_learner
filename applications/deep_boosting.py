import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# flatten images
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.reshape[0], num_pixels).asty