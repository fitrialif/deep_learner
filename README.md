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

        pip install -r requirements.pip

## Usage

Each module can be used for a specific purpose. As an example, the `transfer_learning` module is used for transfer learning (pretty creative, I know). Just specify the location of your `data` folder in the arguments, and you are good to go.

Check the [wiki](https://github.com/anhelus/deep_learner/wiki) for all the necessary information.

## Contributing

Learn how to [contribute](https://github.com/anhelus/deep_learner/blob/master/contributing.md).

## Credits

Angelo Cardellicchio 2017

## License

The project is released under [MIT](https://opensource.org/licenses/MIT) license.
