# import packages
from keras.applciations import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-model", "--model", type=str, default="vgg16", help="name of pre-trained network to use")
ap.add_argument("-iw", "--image_width", type=int, default=60, help="image width")
ap.add_argument("-ih", "--image_height", type=int, default=116, help="image height")
args = vars(ap.parse_args())

MODELS = {
	"vgg16": VGG16,
	"vgg19": VGG19,
	"inception": InceptionV3,
	"xception": Xception,
	"resnet": ResNet50
}


# ensure valid model name
if args["model"] not in MODELS.keys():
	raise AssertionError("The --model command line argument should be a key in the MODELS dictionary")

input_shape = (args["ih"], args["iw"])
preprocess = imagenet_utils.preprocess_input

# if we are using InceptionV3 or Xception
if args["model"] in ("inception", "xception"):
	input_shape = (round(args["ih"]*3), round(args["iw"]*3))
	preprocess = preprocess_input


