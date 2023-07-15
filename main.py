from tensorflow.keras.models import load_model
import os
from train import train
from dblur import dblur
from unet import UNet
from unet import psnr
from unet import ClipLayer


def main():
    model_file = "dblur.h5"

    # Check if the model file exists, if yes, load the model
    if os.path.exists(model_file):
        model = load_model(model_file, custom_objects={'psnr': psnr, 'ClipLayer': ClipLayer})
    else:
        # If the model file doesn't exist, create a new UNet model with the custom learning rate
        learning_rate = 0.001
        model = UNet(input_shape=(None, None, 3), learning_rate=learning_rate)

    train(model, model_file)
    dblur(model)


if __name__ == "__main__":
    main()
