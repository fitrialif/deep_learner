import numpy as np
from keras.datasets import cifar100
# TODO: RE-ARRANGE!
import keras
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import scipy
from scipy import misc
import os

#TODO: load data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
y_train = np.squeeze(y_train)
print('data loaded')

model = InceptionV3(weights='imagenet', include_top=False, input_shape=(139, 139, 3))

if os.path.exists('inception_features_train.npz'):
    print('bottleneck features detected (train)')
    features = np.load('incetpion_features_train.npz')['features']
else:
    print('calculating bottleneck features')
    big_x_train = np.array([scipy.misc.imresize(x_train[i], (139, 139, 3)) for i in range(0, len(x_train))]).astype('float32')
    inception_input_train = preprocess_input(big_x_train)
    features = model.predict(inception_input_train)
    features = np.squeeze(features)
    np.savez('inception_features_train', features=features)
print('bottleneck features saved (train)')

# obtain bottleneck features (test)
if os.path.exists('inception_features_test.npz'):
    print('bottleneck features detected (test)')
    features_test = np.load('inception_features_test.npz')['features_test']
else:
    print('bottleneck features file not detected (test)')
    print('calculating now ...')
    # pre-process the test data
    big_x_test = np.array([scipy.misc.imresize(x_test[i], (139, 139, 3)) 
                       for i in range(0, len(x_test))]).astype('float32')
    inception_input_test = preprocess_input(big_x_test)
    # extract, process, and save bottleneck features (test)
    features_test = model.predict(inception_input_test)
    features_test = np.squeeze(features_test)
    np.savez('inception_features_test', features_test=features_test)
print('bottleneck features saved (test)')