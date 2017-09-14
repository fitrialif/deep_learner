import os
import sys
import glob
import argparse
import utils

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

n=1

# Parameters
img_width, img_height = 75*n, 187*n
train_data_dir = "duke_dataset/train"
validation_data_dir = "duke_dataset/test"

def get_nb_files(directory):
    """
    Get the number of files by searching folder recursively.
    
    Arguments:
    - directory: a string with the name of the folder to search recursively.
    """
    if not os.path.exists(directory):
        return 0
    file_count = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            file_count += len(glob.glob(os.path.join(r, dr + "/*")))
    return file_count


def get_nb_classes(train_dir):
    return len(glob.glob(train_dir + "/*"))

def setup_to_transfer_learn(base_model, model):
    """
    Freeze all layers and compile the model
    """
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

def add_new_last_layer(base_model, nb_classes, fc_size):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(fc_size, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    return model

def setup_to_finetune(model, layers_to_freeze):
    for layer in model.layers[:layers_to_freeze]:
        layer.trainable = False
    for layer in model.layers[layers_to_freeze:]:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

def train(args):
    nb_train_samples = get_nb_files(args.train_dir)
    nb_classes = len(glob.glob(args.train_dir + "/*"))
    nb_val_samples = get_nb_files(args.val_dir)
    nb_epochs = int(args.nb_epoch)
    batch_size = int(args.batch_size)

    
    # init train and test generators
    train_datagen = ImageDataGenerator(
        rescale = 1./255,
        horizontal_flip=True,
        fill_mode = 'nearest')

    test_datagen = ImageDataGenerator(
        rescale = 1./255,
        horizontal_flip=True,
        fill_mode='nearest')

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        class_mode='categorical')
    
    tensorboard = TensorBoard(log_dir='./logs')
    early = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=3, verbose=1, mode='auto')

    base_model = applications.VGG16(weights='imagenet', include_top=False)
    model = add_new_last_layer(base_model, get_nb_classes(train_data_dir), 1024)

    setup_to_transfer_learn(base_model, model)

    history_tl = model.fit_generator(
        train_generator,
        nb_epoch=nb_epochs,
        samples_per_epoch=nb_train_samples,
        validation_data=validation_generator,
        nb_val_samples=nb_val_samples,
        class_weight='auto',
        callbacks=[tensorboard, early])
    
    utils.setup_to_finetune(model, 10)

    history_ft = model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epochs,
        validation_data=validation_generator,
        nb_val_samples=nb_val_samples,
        class_weight='auto',
        callbacks=[early])
    
    model.save(args.output_model_file)

    if args.plot:
        plot_training(history_ft)

def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.show()

if __name__=='__main__':
    a = argparse.ArgumentParser()
    a.add_argument("--train_dir", default='data/train')
    a.add_argument("--val_dir", default='data/val')
    a.add_argument("--nb_epoch", default=50)
    a.add_argument("--batch_size", default=128)
    a.add_argument("--output_model_file", default="inceptionv3-ft.model")
    a.add_argument("--plot", action="store_true")

    args = a.parse_args()

    if args.train_dir is None or args.val_dir is None:
        a.print_help()
        sys.exit(1)

    if (not os.path.exists(args.train_dir)) or (not os.path.exists(args.val_dir)):
        print("directories do not exist")
        sys.exit(1)

    train(args)