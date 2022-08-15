import numpy as np
import pandas as pd
import keras
from matplotlib.image import imread
import ShipDetection

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, listIDs, masksEncoded, batch_size=32, dim=(768,768,3), shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.masksEncoded = masksEncoded
        self.listIDs = listIDs
        self.shuffle = shuffle
        self.on_epoch_end()
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.listIDs) / self.batch_size))
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.listIDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.listIDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, listIDsTemp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim[0],self.dim[1],self.dim[2]))
        y = np.empty((self.batch_size, self.dim[0],self.dim[1],self.dim[2]))
        # Generate data
        for i, ID in enumerate(listIDsTemp):
            # Store sample
            X[i] = imread(ID)
            a = ID[-13:]
            y_enc = self.masksEncoded.loc[a].astype("str")
            y_decoded = ShipDetection.decode(y_enc,self.dim[0],self.dim[1])
            y[i] = X[i].copy()
            for k in range(self.dim[0]):
                for j in range(self.dim[1]):
                    if y_decoded[k][j] == 1:
                        y[i][k][j][0] = 255
        return X, y