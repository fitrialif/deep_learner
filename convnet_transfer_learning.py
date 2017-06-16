from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense

weights_path = '/pretrainednetworks/vgg16_weights'
img_width, img_height = 116, 116

train_data_dir = 'data/train'
test_data_dir = 'data/test'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16

# build vgg16 network
model = applications.VGG16(weights='imagenet', include_top=False)