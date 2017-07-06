# TODO: preprocessing
from keras.applications import ResNet50, InceptionV3, Xception, VGG16, VGG19
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, CSVLogger

from utils.image_utils import *
from utils.transfer_learning_utils import *
from utils.plot_utils import *

import argparse

#TODO: complete description and parsing
ap = argparse.ArgumentParser()
ap.add_argument("-tr", "--traindir", default="../data/train", help="path to the training dataset")
ap.add_argument("-ts", "--testdir", default="../data/test", help="path to the testing dataset")
ap.add_argument("-m", "--model", default="vgg16", help="pre-trained network to use")
ap.add_argument("-mn", "--modelname")
ap.add_argument("-s", "--save", default=True)
ap.add_argument("-sd", "--savedir", default="../models")
args = vars(ap.parse_args())


def check_parameters(args):
    models = {
        "vgg16": VGG16,
        "vgg19": VGG19,
        "inception": InceptionV3,
        "xception": Xception,
        "resnet": ResNet50
    }
    # Check directories.
    if not os.path.exists(os.path.abspath(args["traindir"])):
        raise AssertionError("Training directory does not exist.")
    if not os.path.exists(os.path.abspath(args["testdir"])):
        raise AssertionError("Testing directory does not exist.")
    if not os.path.exists(os.path.abspath(args["savedir"])):
        raise AssertionError("Save directory does not exist.")
    if args["model"] not in models.keys():
        raise AssertionError("The --model command line argument must be a key in the models dictionary.")


def setup(arguments):
    # Check parameters.
    check_parameters(arguments)
    train_dir = args["traindir"]
    test_dir = args["testdir"]
    nb_epoch = 10
    batch_size = 128
    fc_size = 256

    if args["model"] in ("inception", "xception"):
        # TODO: Reshape images if necessary
        img_width = 60*2.32
        img_height = 60*2.32
        LAYERS_TO_FREEZE = 249
    else:
        img_width = 60
        img_height = 160
        LAYERS_TO_FREEZE = 5


def train(train_dir, test_dir, input_shape, layers_to_freeze):
    nb_train_samples = get_nb_files(train_dir)
    nb_classes = len(glob.glob(train_dir + "/*"))
    nb_val_samples = get_nb_files(test_dir)
    # Prepare training and testing data.
    train_data_gen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    test_data_gen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    # Data generators.
    train_generator = train_data_gen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size)
    test_generator = test_data_gen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size)
    # setup model
    base_model = VGG16(weights="imagenet",
                       include_top=False,
                       input_shape=(img_width, img_height, 3))
    model = replace_classification_layer(base_model, 256, nb_classes)
    # Transfer learning.
    setup_transfer_learn(base_model, model)
    early_stopping = EarlyStopping(monitor='val_loss',
                                   min_delta=0.01,
                                   patience=2,
                                   verbose=0,
                                   mode='auto')
    csv_logger_tl = CSVLogger('./logs/tr_learn_reid_vgg16.log')
    csv_logger_ft = CSVLogger('./logs/fine_tun_reid_vgg16.log')
    model_checkpoint = ModelCheckpoint('./checkpoints/')
    history_tl = model.fit_generator(train_generator,
                                     nb_epoch=25,
                                     samples_per_epoch=nb_train_samples,
                                     validation_data=test_generator,
                                     nb_val_samples=nb_val_samples,
                                     class_weight='auto',
                                     callbacks=[early_stopping, csv_logger_tl])
    plot_history_results(history_tl)
    # Fine tuning.
    setup_fine_tuning(model, layers_to_freeze)
    history_ft = model.fit_generator(train_generator,
                                     samples_per_epoch=nb_train_samples,
                                     nb_epoch=50,
                                     validation_data=test_generator,
                                     nb_val_samples=nb_val_samples,
                                     class_weight='auto',
                                     callbacks=[early_stopping, csv_logger_ft])
    plot_history_results(history_ft)

    model.save('models/vgg16_reid.h5')