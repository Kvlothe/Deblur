import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

from loader import load_images
from unet import StepDecay


def train(model, model_file):
    # Load your new training data
    train_blur = 'Train/blur1'
    train_sharp = 'Train/sharp1'
    # train_blur = 'Gopro/train/blur'
    # train_sharp = 'Gopro/train/sharp'
    test_blur = 'Test/blur1'
    test_sharp = 'Test/sharp1'
    # test_blur = 'validation/blur'
    # test_sharp = 'validation/sharp'
    # test_blur = 'Gopro/test/blur'
    # test_sharp = 'Gopro/test/sharp'

    blur_train_images = load_images(train_blur)
    sharp_train_images = load_images(train_sharp)
    blur_test_images = load_images(test_blur)
    sharp_test_images = load_images(test_sharp)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        interpolation_order=3,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='reflect'
    )

    batch_size = 1
    epoch_size = 1

    # Use the generator to create an iterator for your dataset
    train_iterator = datagen.flow(blur_train_images, sharp_train_images, batch_size=batch_size)

    # Instantiate the early stopping callback with desired parameters
    # Change stop rate, over fitting and under fitting **EDIT TO FINE TUNE**
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=10,
                                   verbose=1,
                                   restore_best_weights=True)

    # Instantiate the step decay callback with the desired parameters
    # Change learning rate, **EDIT TO FINE TUNE**
    # learning_rate = 0.0001
    learning_rate = 0.01
    step_decay_callback = StepDecay(initial_lr=learning_rate,
                                    decay_factor=0.5,
                                    step_size=10)

    # Train the model on new data
    # history = model.fit(train_iterator,
    #                     validation_data=(blur_test_images, sharp_test_images),
    #                     epochs=epoch_size,
    #                     # batch_size=batch_size,
    #                     callbacks=[early_stopping, step_decay_callback])

    history = model.fit(blur_train_images, sharp_train_images,
                        validation_data=(blur_test_images, sharp_test_images),
                        epochs=epoch_size,
                        batch_size=batch_size,
                        callbacks=[early_stopping, step_decay_callback])

    # Save the updated model
    model.save(model_file)
    # print(history.history.keys())
    plot_training_history(history)


def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot training and validation loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # Plot training and validation MAE
    # axes[1].plot(history.history['mae'], label='Training MAE')
    # axes[1].plot(history.history['val_mae'], label='Validation MAE')
    # axes[1].set_title('Mean Absolute Error')
    # axes[1].set_xlabel('Epochs')
    # axes[1].set_ylabel('MAE')
    # axes[1].legend()

    # axes[1].plot(history.history['psnr'], label='Training PSNR')
    # axes[1].plot(history.history['val_psnr'], label='Validation PSNR')
    # axes[1].set_title('Peak Signal-to-Noise Ratio')
    # axes[1].set_xlabel('Epochs')
    # axes[1].set_ylabel('PSNR')
    # axes[1].legend()

    plt.show()
