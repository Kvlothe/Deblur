from abc import ABC
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, concatenate, Dropout, Input


# Custom class for layer clipping
class ClipLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ClipLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.clip_by_value(inputs, 0.0, 1.0)


class UNet(tf.keras.models.Model, ABC):
    def __init__(self, input_shape=(None, None, 3)):
        super(UNet, self).__init__()

        inputs = Input(shape=input_shape)

        # Define the U-Net model
        # Down sampling
        inputs = keras.layers.Input(shape=(None, None, 3))
        conv1 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
        conv1 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(conv1)
        pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(pool1)
        conv2 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv2)
        pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool2)
        conv3 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv3)
        pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool3)
        conv4 = keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv4)
        pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = keras.layers.Conv2D(512, 3, activation='relu', padding='same')(pool4)
        conv5 = keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv5)

        up6 = keras.layers.concatenate([keras.layers.UpSampling2D(size=(2, 2))(conv5), conv4], axis=-1)
        conv6 = keras.layers.Conv2D(256, 3, activation='relu', padding='same')(up6)
        conv6 = keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv6)

        up7 = keras.layers.concatenate([keras.layers.UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)
        conv7 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(up7)
        conv7 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv7)

        up8 = keras.layers.concatenate([keras.layers.UpSampling2D(size=(2, 2))(conv7), conv2], axis=-1)
        conv8 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(up8)
        conv8 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv8)

        up9 = keras.layers.concatenate([keras.layers.UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
        conv9 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(up9)
        conv9 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(conv9)

        outputs = keras.layers.Conv2D(3, 1, activation='sigmoid')(conv9)

        super(UNet, self).__init__(inputs=inputs, outputs=outputs)

    def compile(self, optimizer='adam', loss='mean_squared_error', **kwargs):
        super(UNet, self).compile(optimizer=optimizer, loss=loss, **kwargs)
