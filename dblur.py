import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt


def dblur(model):
    image_size = (1024, 1024)
    # Load a blurry image
    img = Image.open('3.jpg')
    # original_size = img.shape
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

    # Convert from BGR to RGB
    deblurred_img = cv2.cvtColor(deblurred_img, cv2.COLOR_BGR2RGB)

    # img = img.resize(original_size)
    # Save the deblurred image
    Image.fromarray(deblurred_img).save('deblurred.png')

    # Display the deblurred image
    plt.imshow(deblurred_img)
    plt.show()
