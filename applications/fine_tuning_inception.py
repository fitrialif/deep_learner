from keras.applications.inception_v3 import InceptionV3

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras import backend as K

from utils import ImageReadUtils



# 
img_width = 60
img_height = 160

input_tensor = Input(shape=(img_width, img_height, 3))


# train and test dirs
train_dir = 'data-reid/train'
test_dir = 'data-reid/test'

nb_train_samples = ImageReadUtils.get_nb_files(train_dir)
nb_val_samples = ImageReadUtils.get_nb_files(test_dir)

# create the base pre-trained
base_model = InceptionV3(weights='imagenet',
	include_top=False,
	input_tensor = input_tensor)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# add a logistc layer
predictions = Dense(183, activation='softmax')(x)

#this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# train only the top layers (i.e. freeze all convolutional InceptionV3 layers)
for layer in base_model.layers:
	layer.trainable = False

# compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# prepare data
train_datagen = ImageDataGenerator(
	rotation_range=30,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True)

test_datagen = ImageDataGenerator(
	rotation_range=30,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
	train_dir,
	target_size=(img_width, img_height),
	batch_size=128)

test_generator = train_datagen.flow_from_directory(
	test_dir,
	target_size=(img_width, img_height),
	batch_size=128)

# train the model
history = model.fit_generator(
	train_generator,
	samples_per_epoch=nb_train_samples,
	nb_epoch=10,
	validation_data=test_generator,
	nb_val_samples=nb_val_samples,
	class_weight='auto')

plotHistoryResults(history)

# the top layers are well trained. start fine tuning conv layers from inception v3 freeze N bottom layers, train remaining top layers

for layer in model.layers[:249]:
	layer.trainable = False
for layer in model.layers[249:]:
	layer.trainable = True


from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')


history_fl = model.fit_generator(
	train_generator,
	samples_per_epoch=nb_train_samples,
	nb_epoch=nb_epoch,
	validation_data=validation_generator,
	nb_val_samples=nb_val_samples,
	class_weight='auto')