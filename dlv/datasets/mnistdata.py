import numpy as np

from keras.utils import np_utils
from keras import backend as K
from keras.datasets import mnist as keras_mnist

from dlv.datasets.dataset import Dataset


# ------------------------------------------------------------------------------

class MnistData(Dataset):
    """A DLV-supported representation of the MNIST dataset.
    """

    name = 'mnist'
    N_CHANNELS = 1
    N_CLASSES = 10

    def __init__(self):
        X_train, Y_train, X_test, Y_test = self.load()
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

# ------------------------------------------------------------------------------

    def load(self):
        img_size = 28

        (X_train, y_train), (X_test, y_test) = keras_mnist.load_data()

        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
        if K.backend() == 'tensorflow':
            X_train = X_train.reshape(n_train, img_size, img_size,
                                      self.N_CHANNELS)
            X_test = X_test.reshape(n_test, img_size, img_size,
                                    self.N_CHANNELS)
        else:

            X_train = X_train.reshape(n_train, self.N_CHANNELS, img_size,
                                      img_size)
            X_test = X_test.reshape(n_test, self.N_CHANNELS, img_size, img_size)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255.
        X_test /= 255.
        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')

        # Convert class vectors to binary class matrices.
        Y_train = np_utils.to_categorical(y_train, self.N_CLASSES)
        Y_test = np_utils.to_categorical(y_test, self.N_CLASSES)

        return X_train, Y_train, X_test, Y_test

# ------------------------------------------------------------------------------

    def getTestImage(self, indx):
        """Return image.
        """
        image = self.X_test[indx:indx + 1]
        if K.backend() == 'tensorflow':
            return image[0]
        else:
            return np.squeeze(image)

# ------------------------------------------------------------------------------

    def labels(self, index):
        return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'][index]
