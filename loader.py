import os
import numpy as np
from PIL import Image
from PIL import ImageOps

# Define the size of your input images
image_size = (512, 512)


# Resize image by a ratio

def resize_image_by_ratio(image, ratio):
    # Get the original dimensions of the image
    width, height = image.size

    # Calculate new dimensions by multiplying the original dimensions by the ratio
    new_width = int(width * ratio)
    new_height = int(height * ratio)

    # Resize the image using the new dimensions
    resized_image = image.resize((new_width, new_height))

    return resized_image


# Resize by ratio
def resize_by_ratio(image, ratio):
    # Get the original dimensions of the image
    width, height = image.size

    # Calculate new dimensions by multiplying the original dimensions by the ratio
    new_width = int(width * ratio)
    new_height = int(height * ratio)

    # Resize the image using the new dimensions
    new_size = image.resize((new_width, new_height))

    return new_size


def pad_image_to_square(image, fill_value=0):
    width, height = image.size
    max_dimension = max(width, height)

    padded_image = ImageOps.expand(image, (
        (max_dimension - width) // 2,
        (max_dimension - height) // 2,
        (max_dimension - width + 1) // 2,
        (max_dimension - height + 1) // 2
    ), fill=fill_value)

    return padded_image


# Method for loading data
def load_images(image_path):
    image_array = []
    ratio = 0.5
    for filename in os.listdir(image_path):
        img = Image.open(os.path.join(image_path, filename))
        # image_size = resize_image_by_ratio(img, ratio)
        # resized_image = resize_image_by_ratio(img, ratio)
        img = pad_image_to_square(img, fill_value=0)
        img = img.resize(image_size)
        img = np.array(img) / 255
        image_array.append(img)
    image_array = np.array(image_array)
    return image_array
