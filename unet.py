from abc import ABC
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, concatenate, Dropout, Input
from tensorflow.keras.callbacks import Callback


# Class for layer clipping
class ClipLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ClipLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.clip_by_value(inputs, 0.0, 1.0)


# Class Step Decay for fine-tuning
class StepDecay(Callback):
    def __init__(self, initial_lr, decay_factor, step_size):
        super(StepDecay, self).__init__()
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor
        self.step_size = step_size

    def on_epoch_begin(self, epoch, logs=None):
        new_lr = self.initial_lr * (self.decay_factor ** (epoch // self.step_size))
        self.model.optimizer.lr.assign(new_lr)


# U-Net class
class UNet(tf.keras.models.Model, ABC):
    def __init__(self, input_shape=(None, None, 3)):
        super(UNet, self).__init__()

        inputs = Input(shape=input_shape)

        # Define the U-Net model
        # Down sampling
        inputs = keras.layers.Input(shape=(None, None, 3))
        conv1 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        pool1 = Dropout(0.5)(pool1)

        conv2 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        pool2 = Dropout(0.5)(pool2)

        conv3 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
        pool3 = Dropout(0.5)(pool3)

        conv4 = keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
        pool4 = Dropout(0.5)(pool4)

        conv5 = keras.layers.Conv2D(512, 3, activation='relu', padding='same')(pool4)
        conv5 = BatchNormalization()(conv5)
        conv5 = keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = Dropout(0.5)(conv5)

        # up sampling
        up6 = keras.layers.concatenate([keras.layers.UpSampling2D(size=(2, 2))(conv5), conv4], axis=-1)
        conv6 = keras.layers.Conv2D(256, 3, activation='relu', padding='same')(up6)
        conv6 = BatchNormalization()(conv6)
        conv6 = keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv6)
        conv6 = BatchNormalization()(conv6)

        up7 = keras.layers.concatenate([keras.layers.UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)
        conv7 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(up7)
        conv7 = BatchNormalization()(conv7)
        conv7 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv7)
        conv7 = BatchNormalization()(conv7)

        up8 = keras.layers.concatenate([keras.layers.UpSampling2D(size=(2, 2))(conv7), conv2], axis=-1)
        conv8 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(up8)
        conv8 = BatchNormalization()(conv8)
        conv8 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv8)
        conv8 = BatchNormalization()(conv8)

        up9 = keras.layers.concatenate([keras.layers.UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
        conv9 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(up9)
        conv9 = BatchNormalization()(conv9)
        conv9 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(conv9)
        conv9 = BatchNormalization()(conv9)

        clip = ClipLayer()(conv9)
        outputs = keras.layers.Conv2D(3, 1, activation='sigmoid')(clip)

        super(UNet, self).__init__(inputs=inputs, outputs=outputs)

    def compile(self, optimizer='adam', loss='mean_squared_error', **kwargs):
        super(UNet, self).compile(optimizer=optimizer, loss=loss, **kwargs)
