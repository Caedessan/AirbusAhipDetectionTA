import DataGenerator
import ShipDetection
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Input
from keras.optimizers import Adam
import pandas as pd
import glob



import tensorflow as tf

with tf.device("/CPU:0"):
    tf.debugging.set_log_device_placement(True)


    # Set local variables

    imgWidth = 256
    imgHeight = 256
    batch_size = 8
    trainDir = "data/train/"
    validDir = "data/validation/"


    # Set paths to data

    trainPaths = glob.glob(trainDir + "*.jpg")
    validationPaths = glob.glob(validDir + "*.jpg")

    train_ship_segmentations_v2 = pd.read_csv("train_ship_segmentations_v2.csv")
    train_ship_segmentations_v2.set_index("ImageId", inplace=True)

    # Create Data generators for model

    training_generator = DataGenerator.DataGenerator(trainPaths,
                                                     train_ship_segmentations_v2,
                                                     batch_size=batch_size)
    validation_generator = DataGenerator.DataGenerator(validationPaths,
                                                       train_ship_segmentations_v2,
                                                       batch_size=batch_size)


    # Create model

    inputImg = Input((imgWidth, imgHeight, 3), name='img')
    model = ShipDetection.get_unet(inputImg)
    model.compile(optimizer=Adam(), loss=ShipDetection.dice_coef_loss)

    # Set fitting callbacks

    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
        ModelCheckpoint('model1', verbose=1, save_best_only=True, save_weights_only=True)
    ]


    # Fit model

    results = model.fit(training_generator,
                        validation_data=validation_generator,
                        epochs=5,
                        callbacks=callbacks,
                        verbose=1
                        )

