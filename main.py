from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
from loader import load_images
from unet import UNet
import numpy as np
from PIL import Image
import cv2
from unet import StepDecay

image_size = (256, 256)


def main():
    model_file = "dblur.h5"

    # Check if the model file exists, if yes, load the model
    if os.path.exists(model_file):
        model = load_model(model_file)
    else:
        # If the model file doesn't exist, create a new UNet model
        model = UNet(input_shape=(None, None, 3))

        # Set custom learning rate
        learning_rate = 0.0001
        optimizer = Adam(learning_rate=learning_rate)

        # Compile the model using the custom optimizer
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        # model.compile(optimizer='adam', loss='mean_squared_error')

    # Load your new training data
    x_load = 'Gopro/train/Blur'
    y_load = 'Gopro/train/Sharp'
    # x_load = 'C:/Users/komun/Downloads/archive/DBlur/Wider-Face/train/blur'
    # y_load = 'C:/Users/komun/Downloads/archive/DBlur/Wider-Face/train/sharp'
    x_test = 'Gopro/test/Blur'
    y_test = 'Gopro/test/Sharp'
    # x_test = 'C:/Users/komun/Downloads/archive/DBlur/Wider-Face/test/blur'
    # y_test = 'C:/Users/komun/Downloads/archive/DBlur/Wider-Face/test/sharp'

    x_train_images = load_images(x_load)
    y_train_images = load_images(y_load)
    x_test_images = load_images(x_test)
    y_test_images = load_images(y_test)

    # Adding a way to stop the training early if the value loss does not decrease
    # Change stop rate, over fitting and under fitting **EDIT TO FINE TUNE**
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=5,
                                   verbose=1,
                                   restore_best_weights=True)

    # Instantiate the step decay callback with the desired parameters
    # Change learning rate, **EDIT TO FINE TUNE**
    step_decay_callback = StepDecay(initial_lr=learning_rate, decay_factor=0.5, step_size=10)

    # Train the model on new data
    history = model.fit(x_train_images, y_train_images,
                        validation_data=(x_test_images, y_test_images),
                        epochs=50,
                        batch_size=64,
                        callbacks=[early_stopping, step_decay_callback])

    # Save the updated model
    model.save(model_file)

    # Load a blurry image
    img = Image.open('3.jpg')
    # original_size = img.shape
    # Resize the image to the desired input size
    # img = img.resize(image_size)

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


if __name__ == "__main__":
    main()
