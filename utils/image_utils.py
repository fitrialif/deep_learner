import os
import glob
import numpy as np

def arrange_data_set(directory, file_extension='*.jpg'):
    if not os.path.exists(directory):
        return 0
    cur_dir = os.getcwd()
    os.chdir(directory)
    work_dir = os.getcwd()
    file_list = glob.glob('*.jpg')
    train_dir = os.path.abspath('train')
    test_dir = os.path.abspath('test')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    os.chdir(cur_dir)
    init_label = file_list[0][0:4]
    current_list = []
    for file in file_list:
        # TODO: param label with regex and boundaries
        label = file[0:4]
        # if label is the same, append to list, else evolve list
        if label == init_label:
            current_list.append(file)
        if (label != init_label) or (file == file_list[-1]):
            if(len(current_list) > 10):
                if not os.path.exists(os.path.join(train_dir, init_label)):
                    os.mkdir(os.path.join(train_dir, init_label))
                if not os.path.exists(os.path.join(test_dir, init_label)):
                    os.mkdir(os.path.join(test_dir, init_label))
                partial_train, partial_test = arrange_dataset_2(current_list)
                for train_file in partial_train:
                    os.rename(os.path.join(work_dir, train_file), os.path.join(train_dir, init_label, train_file))
                for test_file in partial_test:
                    os.rename(os.path.join(work_dir, test_file), os.path.join(test_dir, init_label, test_file))
            current_list[:] = []
            init_label = label
            current_list.append(file)


def arrange_dataset_2(file_list, train_split=0.8):
    random_set = np.random.permutation(len(file_list))
    train_list = random_set[:round(len(random_set)*train_split)]
    test_list = random_set[-(len(file_list) - len(train_list))::]
    train_images = []
    test_images = []
    for index in train_list:
        train_images.append(file_list[index])
    for index in test_list:
        test_images.append(file_list[index])
    return train_images, test_images


def get_nb_files(directory):
    """ Get the number of files by searching folder recursively."""
    if not os.path.exists(directory):
        return 0
    file_count = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            file_count += len(glob.glob(os.path.join(r, dr + "/*")))
    return file_count


def get_nb_classes(train_dir):
    return len(glob.glob(train_dir + "/*"))