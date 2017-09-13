import os
import glob
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split


def set_image_format(img_rows, img_cols, img_channels, keras_backend):
    """
    """
    if keras_backend == 'channels_first':
        input_shape = (img_channels, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, img_channels)


def get_nb_files(root_folder):
    """Get the number of files in a folder's hierarchy.
    :param root_folder: A string which represents the root folder of the hierarchy.
    :return file_count: An integer which represents the number of files in the folder's hierarchy.
    """
    if not os.path.exists(root_folder):
        return 0
    file_count = 0
    for r, dirs, files in os.walk(root_folder):
        for dr in dirs:
            file_count += len(glob.glob(os.path.join(r, dr + "/*")))
    return file_count


def get_nb_classes(folder):
    """Get the number of classes in a folder.
    :param folder: A string which represents the main data folder.
    :return class_count: An integer which represents the number of classes in the main data folder.
    """
    if not os.path.exists(folder):
        return 0
    class_count = len(glob.glob(folder + "/*"))
    return class_count


def setup_transfer_learning(base_model, model):
    """Freeze layers from the base_model and compile the model.
    :param base_model: The part of the model that will not be re-trained for transfer learning.
    :param model: The 
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


def split_each_label(root_folder, train_folder, test_folder):
    # train dir = os.path.abspath('train')
    # test_dir = os.path.abspath('test')
    # try:
    # os.makedirs(train_dir)
    # os.makedirs(test_dir)
    # except OSError:
    # pass
    if not os.path.exists(root_folder):
        return 0
    for root, dirs, files in os.walk(root_folder):
        for d in dirs:
            act_len = len(glob.glob(os.path.join(root, d + "/*")))
            dir_path = os.path.join(os.path.relpath(d, root_folder))
            if act_len > 20:
                x = y = os.listdir(dir_path)
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
                for x in x_train:
                    os.rename(os.path.join(dir_path, x), os.path.join(train_root_folder + x, train_folder + x)
                for x in x_test:
                    os.rename(root_folder + x, test_folder + x)