from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import os
from train import train
from dblur import dblur
from unet import UNet


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

    train(model, model_file)
    dblur(model)


if __name__ == "__main__":
    main()
