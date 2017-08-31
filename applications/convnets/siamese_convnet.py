from __future__ import division, print_function

from keras import backend as K
from keras.applications import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, merge
from keras.layers.core import Activation, Dense, Dropout, Lambda
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from random import shuffle
from scipy.misc import imresize
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os

def pick_random_image(train_dir):
    """
    Pick a random image from the dataset.
    """
    # 1. Get the absolute path of train directory.
    choose_dir = os.path.join(os.getcwd(), train_dir)
    # 2. Randomly pick an image group.
    image_group = os.path.join(choose_dir, random.choice(os.listdir(choose_dir)))
    # 3. Randomply pick an image from the image group.
    image_pick = os.path.join(image_group, random.choice(os.listdir(image_group)))
    return image_group, image_pick