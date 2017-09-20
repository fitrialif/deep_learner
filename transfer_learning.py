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


def transfer_learning(base_model, model, model_load, optimizer=optimizers.SGD(lr=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy']):
    """
    Freeze layers from the base model.
    
    :param base_model: The part of the model that will not be re-trained for transfer learning.
    :param model: The 
    """
    if model_load:
        model.load_weights(model_load)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics)


def replace_classification_layer(base_model, nb_classes, fc_size):
    """
    Replace the classification layer in a pre-trained model. Uses the Keras Functional API.
    
    Arguments:

    Return:

    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(fc_size, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    return model


def setup_fine_tuning(model, layers_to_freeze, model_load):
    """
    Setup the network for fine tuning.
    The first layers_to_freeze layers will be made not trainable, while the others will be fine-tuned on the novel dataset.
    Arguments:

    Return:

    """
    for layer in model.layers[:layers_to_freeze]:
        layer.trainable = False
    for layer in model.layers[layers_to_freeze:]:
        layer.trainable = True
    model.compile(optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


def train(args):
    """
    Performs training.
    """
    train_dir = args.train_folder
    val_dir = args.validation_folder
    nb_train_samples = utils.get_nb_files(args.train_folder)
    nb_val_samples = utils.get_nb_files(args.validation_folder)
    nb_classes = utils.get_labels(args.train_folder, args.validation_folder)
    nb_epochs = int(args.nb_epoch)
    batch_size = int(args.batch_size)
    base_architecture = args.base_architecture
    model_load = args.model_load
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
        base_model = applications.Xception(weights='imagenet', include_top=False)
    model = replace_classification_layer(base_model, nb_classes, 1024)
    
    # Data augmentation.
    train_datagen = ImageDataGenerator(
        rescale = 1./255,
        horizontal_flip=True,
        fill_mode = 'nearest')
    test_datagen = ImageDataGenerator(
        rescale = 1./255,
        horizontal_flip=True,
        fill_mode='nearest')
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(img_height, img_width),
        class_mode='categorical')
    

    transfer_learning(base_model, model, model_load)

    history_tl = model.fit_generator(
        train_generator,
        nb_epoch=nb_epochs,
        samples_per_epoch=nb_train_samples,
        validation_data=validation_generator,
        nb_val_samples=nb_val_samples,
        class_weight='auto',
        callbacks=args.callbacks)
    
    utils.plot_training(history_tl)

    setup_fine_tuning(model, layers_to_freeze, model_load)

    history_ft = model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epochs,
        validation_data=validation_generator,
        nb_val_samples=nb_val_samples,
        class_weight='auto',
        callbacks=args.callbacks)
    
    # NOTE
    model.save(os.path.join(os.getcwd(), 'models', args.output_model_file))

    utils.plot_training(history_ft)



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