import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import cv2

# Define the U-Net model
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

model = keras.models.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='mean_squared_error')

########################################
# Define the path to your blurry and clear test image datasets
blurry_test_images_path = 'Gopro/test/blur'
clear_test_images_path = 'Gopro/test/sharp'
# Define the path to your blurry and clear image datasets
blurry_images_path = 'Gopro/train/blur'
clear_images_path = 'Gopro/train/sharp'

# Define the size of your input and output images
image_size = (256, 256)

# Load the blurry test images into a NumPy array
X_test = []
for filename in os.listdir(blurry_test_images_path):
    img = Image.open(os.path.join(blurry_test_images_path, filename))
    img = img.resize(image_size)
    img = np.array(img)
    X_test.append(img)
X_test = np.array(X_test)

# Load the clear test images into a NumPy array
y_test = []
for filename in os.listdir(clear_test_images_path):
    img = Image.open(os.path.join(clear_test_images_path, filename))
    img = img.resize(image_size)
    img = np.array(img)
    y_test.append(img)
y_test = np.array(y_test)
###################################

# Load the blurry test images into a NumPy array
X_test = []
for filename in os.listdir(blurry_test_images_path):
    img = Image.open(os.path.join(blurry_test_images_path, filename))
    img = img.resize(image_size)
    img = np.array(img)
    X_test.append(img)
X_test = np.array(X_test)
###########################################
# Load the blurry images into a NumPy array
x_train = []
for filename in os.listdir(blurry_images_path):
    img = Image.open(os.path.join(blurry_images_path, filename))
    img = img.resize(image_size)
    img = np.array(img)
    x_train.append(img)
x_train = np.array(x_train)

# Load the clear images into a NumPy array
y_train = []
for filename in os.listdir(clear_images_path):
    img = Image.open(os.path.join(clear_images_path, filename))
    img = img.resize(image_size)
    img = np.array(img)
    y_train.append(img)
y_train = np.array(y_train)
##########################################
# Load the blurry images into a NumPy array
x_train = []
for filename in os.listdir(blurry_images_path):
    img = Image.open(os.path.join(blurry_images_path, filename))
    img = img.resize(image_size)
    img = np.array(img)
    x_train.append(img)
x_train = np.array(x_train)
FIT = model.fit(x_train, y_train, validation_data=(X_test, y_test), epochs=10)

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score)

print('Training session complete')

# Load a blurry image
img = cv2.imread('98.jpg')

# Resize the image to the desired input size
img = cv2.resize(img, image_size)

# Add a batch dimension to the image
img = np.expand_dims(img, axis=0)

# Use the trained model to deblur the image
deblurred_img = model.predict(img)

# Remove the batch dimension from the deblurred image
deblurred_img = np.squeeze(deblurred_img, axis=0)

# Display the deblurred image
plt.imshow(deblurred_img)
plt.show()
