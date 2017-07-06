# Deep Learner

## What is Deep Learner?

*Deep Learner* is a set of Python utilities, built on top of Keras and TensorFlow, which can be used for deep learning.

## Is there any specific advantage in using this instead of Keras?

No. Deep Learner is a collection of the applications I developed for deep learning tasks.

It is built on top of a library (Keras) which is built on top of a library (TensorFlow or Theano) which is built on top of a library... you get the point. Therefore, while you can use it if you have a problem which is already solved by one of the applications, if you have a more specific task, you should probably check Keras itself.

## Installation

1. Install Python. Not sure how to do that? Check [this link](https://www.python.org/).
2. Create a new virtual environment. I personally prefer [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/), but it is not mandatory.
3. Activate the new environment, and clone this repo.
    git clone https://github.com/anhelus/keras_convnets.git
4. Install requirements:
    pip install requirements.pip
5. Create, in the folder where the repository has been cloned, a folder for your train examples (e.g. **data/train**) and a folder for your test examples (e.g. **data/test**).
6. Move to the application folder, and run an application passing the required arguments from the command line.

## Applications

For a complete list of the applications, check the [wiki](https://github.com/anhelus/deep_learner/wiki) (UNDER CONSTRUCTION!).
