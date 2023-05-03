import os
import numpy as np
from PIL import Image

# Define the size of your input images
image_size = (256, 256)


def for_test(image_path):
    test_array = []
    for filename in os.listdir(image_path):
        img = Image.open(os.path.join(image_path, filename))
        img = img.resize(image_size)
        img = np.array(img) / 255
        test_array.append(img)
    test_array = np.array(test_array)
    return test_array


def for_train(image_path):
    train_array = []
    for filename in os.listdir(image_path):
        img = Image.open(os.path.join(image_path, filename))
        img = img.resize(image_size)
        img = np.array(img) / 255
        train_array.append(img)
    train_array = np.array(train_array)
    return train_array

