from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras import backend as K


# from tensorflow.python import padding_fifo_queue


class ShallowNet:
    @staticmethod
    def built(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
