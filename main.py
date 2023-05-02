import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import cv2


class ClipLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ClipLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.clip_by_value(inputs, 0.0, 1.0)


# Define the U-Net model
inputs = keras.layers.Input(shape=(None, None, 3))
conv1 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
conv1 = BatchNormalization()(conv1)
conv1 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(conv1)
conv1 = BatchNormalization()(conv1)
pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(pool1)
conv2 = BatchNormalization()(conv2)
conv2 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv2)
conv2 = BatchNormalization()(conv2)
pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool2)
conv3 = BatchNormalization()(conv3)
conv3 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv3)
conv3 = BatchNormalization()(conv3)
pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool3)
conv4 = BatchNormalization()(conv4)
conv4 = keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv4)
conv4 = BatchNormalization()(conv4)
pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = keras.layers.Conv2D(512, 3, activation='relu', padding='same')(pool4)
conv5 = BatchNormalization()(conv5)
conv5 = keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv5)
conv5 = BatchNormalization()(conv5)
conv5 = Dropout(0.5)(conv5)

up6 = keras.layers.concatenate([keras.layers.UpSampling2D(size=(2, 2))(conv5), conv4], axis=-1)
conv6 = keras.layers.Conv2D(256, 3, activation='relu', padding='same')(up6)
conv6 = BatchNormalization()(conv6)
conv6 = keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv6)
conv6 = BatchNormalization()(conv6)
conv6 = Dropout(0.5)(conv6)

up7 = keras.layers.concatenate([keras.layers.UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)
conv7 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(up7)
conv7 = BatchNormalization()(conv7)
conv7 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv7)
conv7 = BatchNormalization()(conv7)
conv6 = Dropout(0.5)(conv6)

up8 = keras.layers.concatenate([keras.layers.UpSampling2D(size=(2, 2))(conv7), conv2], axis=-1)
conv8 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(up8)
conv8 = BatchNormalization()(conv8)
conv8 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv8)
conv8 = BatchNormalization()(conv8)
conv8 = Dropout(0.5)(conv8)

up9 = keras.layers.concatenate([keras.layers.UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
conv9 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(up9)
conv9 = BatchNormalization()(conv9)
conv9 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(conv9)
conv9 = BatchNormalization()(conv9)
conv9 = Dropout(0.5)(conv9)

outputs = keras.layers.Conv2D(3, 1, activation='sigmoid')(conv9)

model = keras.models.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='mean_squared_error')

########################################
# Define the path to your blurry and clear test image datasets
# blurry_test_images_path = 'Test/Blur'
# clear_test_images_path = 'Test/Sharp'
blurry_test_images_path = 'C:/Users/komun/Downloads/archive/DBlur/Helen/test/blur'
clear_test_images_path = 'C:/Users/komun/Downloads/archive/DBlur/Helen/test/sharp'

# Define the path to your blurry and clear Train image datasets
# blurry_train_images_path = 'Train/Blur'
# clear_train_images_path = 'Train/Sharp'
blurry_train_images_path = 'C:/Users/komun/Downloads/archive/DBlur/Helen/train/blur'
clear_train_images_path = 'C:/Users/komun/Downloads/archive/DBlur/Helen/train/sharp'

# Define the size of your input and output images
image_size = (256, 256)
# image_size = (1280, 720)

# Load the blurry test images into a NumPy array
X_test = []
for filename in os.listdir(blurry_test_images_path):
    # Open image
    img = Image.open(os.path.join(blurry_test_images_path, filename))
    # Resize image
    img = img.resize(image_size)
    # Turn image into a np array
    img = np.array(img)/255.0  # Normalize the image
    # append image to test array
    X_test.append(img)
X_test = np.array(X_test)

# Load the clear test images into a NumPy array
y_test = []
for filename in os.listdir(clear_test_images_path):
    img = Image.open(os.path.join(clear_test_images_path, filename))
    img = img.resize(image_size)
    img = np.array(img)/255.0
    y_test.append(img)
y_test = np.array(y_test)
###################################

###########################################
# Load the blurry images into a NumPy array
x_train = []
for filename in os.listdir(blurry_train_images_path):
    img = Image.open(os.path.join(blurry_train_images_path, filename))
    img = img.resize(image_size)
    img = np.array(img)/255.0
    x_train.append(img)
x_train = np.array(x_train)

# Load the clear images into a NumPy array
y_train = []
for filename in os.listdir(clear_train_images_path):
    img = Image.open(os.path.join(clear_train_images_path, filename))
    img = img.resize(image_size)
    img = np.array(img)/255.0
    y_train.append(img)
y_train = np.array(y_train)
# x_train = x_train.astype('float32') / 255
# y_train = y_train.astype('float32') / 255
# X_test = X_test.astype('float32') / 255
# y_test = y_test.astype('float32') / 255

##########################################
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

# FIT = model.fit(x_train, y_train, validation_data=(X_test, y_test), epochs=50)
history = model.fit(x_train, y_train, validation_data=(X_test, y_test), epochs=50, callbacks=[early_stopping])

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score)


# # Check the range of pixel values for the training dataset
# train_min = np.min(x_train)
# train_max = np.max(x_train)
# print("Training dataset pixel values range: [{}, {}]".format(train_min, train_max))
#
# # Check the range of pixel values for the test dataset
# test_min = np.min(X_test)
# test_max = np.max(X_test)
# print("Test dataset pixel values range: [{}, {}]".format(test_min, test_max))


# Load a blurry image
# img = Image.open('98.png')
# img = Image.open('980.png')
# Load a blurry image
img = Image.open('7.jpg')

# Resize the image to the desired input size
img = img.resize(image_size)

# Convert the image to a numpy array and normalize
img_np = np.array(img) / 255.0

# Convert the image from RGB to BGR (since the model is trained with BGR images)
img_np_uint8 = (img_np * 255).astype('uint8')
img_np_bgr = cv2.cvtColor(img_np_uint8, cv2.COLOR_RGB2BGR) / 255.0

# Use the trained model to deblur the image
deblurred_img = model.predict(np.expand_dims(img_np_bgr, axis=0))

# Remove the batch dimension from the deblurred image
deblurred_img = np.squeeze(deblurred_img, axis=0)

# Convert the deblurred image from floating point values to unsigned 8-bit integers
deblurred_img = np.clip(deblurred_img * 255, 0, 255).astype('uint8')

# Create a new Pillow image from the deblurred image
deblurred_img_pil = Image.fromarray(deblurred_img)

# Convert from BGR to RGB
deblurred_img = cv2.cvtColor(deblurred_img, cv2.COLOR_BGR2RGB)

# Save the deblurred image
Image.fromarray(deblurred_img).save('deblurred.png')

# Display the deblurred image
plt.imshow(deblurred_img)
plt.show()
