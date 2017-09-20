import keras
import numpy as np
import scipy
from scipy import misc
import os

import os
import sys
import glob
import argparse
import utils

from keras import applications
from keras import metrics
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

n=1

# Parameters
img_width, img_height = 75*n, 187*n


def train(args):
    """
    Performs training.
    """
    train_dir = args.train_folder
    nb_train_samples = utils.get_nb_files(args.train_folder)
    nb_classes = utils.get_labels(args.train_folder, args.validation_folder)
    base_architecture = args.base_architecture
    # Define base layer
    if base_architecture == 'VGG16':
        base_model = applications.VGG16(weights='imagenet', include_top=False)
        layers_to_freeze = 10
    elif base_architecture == 'VGG19':
        base_model = applications.VGG19(weights='imagenet', include_top=False)
        layers_to_freeze = 11
    elif base_architecture == 'InceptionV3':
        base_model = applications.InceptionV3(weights='imagenet', include_top=False)
        layers_to_freeze = 172 # TODO: understand how many levels!
    elif base_architecture == 'ResNet50':
        base_model = applications.ResNet50(weights='imagenet', include_top=False)
    
    data_gen = ImageDataGenerator(rescale = 1./255)

    data_generator = data_gen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=1,
        class_mode='categorical'
    )

    data_list = []
    batch_index = 0

    while batch_index <= data_generator.batch_index:
        data = data_generator.next()
        data_list.append(data[0])
        batch_index = batch_index + 1
    
    print(batch_index)
    print(data_generator.batch_index)
    
    data_array = np.asarray(data_list)

    print(data_array)


if __name__=='__main__':
    # Callbacks array
    tensorboard = TensorBoard(log_dir='./logs')
    early = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=3, verbose=1, mode='auto')
    checkpoint = ModelCheckpoint("weights.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks = [tensorboard, early, checkpoint]
    a = argparse.ArgumentParser()
    a.add_argument("--base_architecture", default='InceptionV3')
    a.add_argument("--train_folder", default='data/train')
    a.add_argument("--validation_folder", default='data/test')
    a.add_argument("--nb_epoch", default=1)
    a.add_argument("--batch_size", default=128)
    a.add_argument("--output_model_file", default="inceptionv3.model")
    a.add_argument("--plot", action="store_true")
    a.add_argument("--model_load", default='')
    a.add_argument("--callbacks", default = callbacks)

    args = a.parse_args()

    if args.train_folder is None or args.validation_folder is None:
        a.print_help()
        sys.exit(1)

    if (not os.path.exists(args.train_folder)) or (not os.path.exists(args.validation_folder)):
        print("directories do not exist")
        sys.exit(1)

    train(args)