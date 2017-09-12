from keras.layers import Input, Conv2D, MaxPooling2D
from keras.models import Model


class Siamese:
    

class PseudoSiamese:


class TwoChannels:
    

# Input shape is 6-channels
input_img = Input(shape=(60, 160, 6))

first_nin_1 = Conv2D(32, (1, 1), padding='same', activation='relu')(input_img)
first_nin_2 = Conv2D(32, (1, 1), padding='same', activation='relu')(first_nin_1)

second_nin_1 = Conv2D(32, (1, 1), padding='same', activation='relu')(input_img)
second_nin_2 = Conv2D(32, (5, 5), padding='same', activation='relu')(second_nin_1)

third_nin_1 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
third_nin_2 = Conv2D(32, (5, 5), padding='same', activation='relu')(third_nin_1)

output = keras.layers.concatenate([tower_1, ])