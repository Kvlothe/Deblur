import os
import numpy as np
from PIL import Image

# Define the size of your input images
# image_size = (256, 256)


# Resize image by a ratio - to account for memory issues and dataset mismatch of size
# Also would like to be able to restore original size of image for better training
# Also plan this for future data augmentation
def resize_image_by_ratio(image, ratio):
    # Get the original dimensions of the image
    width, height = image.size

    # Calculate new dimensions by multiplying the original dimensions by the ratio
    new_width = int(width * ratio)
    new_height = int(height * ratio)

    # Resize the image using the new dimensions
    resized_image = image.resize((new_width, new_height))

    return resized_image


# Method for loading data
def load_images(image_path):
    image_array = []
    ratio = 0.5
    for filename in os.listdir(image_path):
        img = Image.open(os.path.join(image_path, filename))
        image_size = resize_image_by_ratio(img, ratio)
        img = img.resize(image_size)
        img = np.array(img) / 255
        image_array.append(img)
    image_array = np.array(image_array)
    return image_array
