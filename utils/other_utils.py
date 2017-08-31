import os
from sklearn.cross_validation import train_test_split
from shutil import copytree

# TODO: algorithm + set into appropriate class
# get the list of current folder
dirs = os.listdir()

def f(list_name):
    X = y = os.listdir(list_name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    for x in X_train:
        copytree(os.path.join(os.getcwd(), list_name, x), os.path.join(os.getcwd(), 'train', x))
    for x in X_test:
        copytree(os.path.join(os.getcwd(), list_name, x), os.path.join(os.getcwd(), 'test', x))

[f(x) for x in dirs]