from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os
from loader import for_train
from loader import for_test
from unet import UNet
import numpy as np
from PIL import Image
import cv2


image_size = (256, 256)


def main():
    model_file = "dblur.h5"

    # Check if the model file exists, if yes, load the model
    if os.path.exists(model_file):
        model = load_model(model_file)
    else:
        # If the model file doesn't exist, create a new UNet model
        model = UNet(input_shape=(None, None, 3))
        model.compile(optimizer='adam', loss='mean_squared_error')

    # Load your new training data
    # x_load = 'Train/Blur'
    # y_load = 'Train/Sharp'
    x_load = 'C:/Users/komun/Downloads/archive/DBlur/Helen/train/blur'
    y_load = 'C:/Users/komun/Downloads/archive/DBlur/Helen/train/sharp'
    x_trainer = for_train(x_load)
    y_trainer = for_train(y_load)
    # x_test = 'Test/Blur'
    # y_test = 'Test/Sharp'
    x_test = 'C:/Users/komun/Downloads/archive/DBlur/Helen/test/blur'
    y_test = 'C:/Users/komun/Downloads/archive/DBlur/Helen/test/sharp'
    x_tester = for_test(x_test)
    y_tester = for_test(y_test)

    # Adding a way to stop the training early if the value loss does not decrease
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=10,
                                   verbose=1,
                                   restore_best_weights=True)

    # Train the model on new data
    history = model.fit(x_trainer, y_trainer,
                        validation_data=(x_tester, y_tester),
                        epochs=25,
                        batch_size=32,
                        callbacks=[early_stopping])

    # Save the updated model
    model.save(model_file)

    # Load a blurry image
    img = Image.open('1.jpg')

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

    # Save the deblurred image
    Image.fromarray(deblurred_img).save('deblurred.png')

    # Display the deblurred image
    plt.imshow(deblurred_img)
    plt.show()


if __name__ == "__main__":
    main()
