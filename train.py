from loader import load_images
from tensorflow.keras.callbacks import EarlyStopping
from unet import StepDecay


def train(model, model_file):
    # Load your new training data
    train_blur = 'Train/blur'
    train_sharp = 'Train/sharp'
    # train_blur = 'Gopro/train/blur'
    # train_sharp = 'Gopro/train/sharp'
    test_blur = 'Test/blur'
    test_sharp = 'Test/sharp'
    # test_blur = 'Gopro/test/blur'
    # test_sharp = 'Gopro/test/sharp'

    blur_train_images = load_images(train_blur)
    sharp_train_images = load_images(train_sharp)
    blur_test_images = load_images(test_blur)
    sharp_test_images = load_images(test_sharp)

    # Instantiate the early stopping callback with desired parameters
    # Change stop rate, over fitting and under fitting **EDIT TO FINE TUNE**
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=5,
                                   verbose=1,
                                   restore_best_weights=True)

    # Instantiate the step decay callback with the desired parameters
    # Change learning rate, **EDIT TO FINE TUNE**
    learning_rate = 0.0001
    step_decay_callback = StepDecay(initial_lr=learning_rate,
                                    decay_factor=0.5,
                                    step_size=10)

    # Train the model on new data
    history = model.fit(blur_train_images, sharp_train_images,
                        validation_data=(blur_test_images, sharp_test_images),
                        epochs=5,
                        batch_size=16,
                        callbacks=[early_stopping, step_decay_callback])

    # Save the updated model
    model.save(model_file)
