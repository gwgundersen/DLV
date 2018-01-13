import scipy.io as sio

from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K
import numpy as np

from dlv.nn.nn import NN


# ------------------------------------------------------------------------------

class MnistNN(NN):
    DIRECTORY = 'dlv/nn/mnist'

    def __init__(self):
        """Load MNIST neural network.
        """
        self.define()
        self.load_weights()
        self.model.summary()

# ------------------------------------------------------------------------------

    def define(self):
        """Define neural network model.
        """
        n_classes = 10
        img_size = 28
        n_filters = 32
        pooling_size = 2
        kernel_size = 3

        if K.backend() == 'tensorflow':
            K.set_learning_phase(0)

        model = Sequential()

        model.add(Convolution2D(n_filters, kernel_size, kernel_size,
                                border_mode='valid',
                                input_shape=(1, img_size, img_size)))
        model.add(Activation('relu'))
        model.add(Convolution2D(n_filters, kernel_size, kernel_size))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(pooling_size, pooling_size)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(n_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])

        self.model = model

# ------------------------------------------------------------------------------

    def predict(self, input_):
        if len(input_.shape) == 2 and K.backend() == 'tensorflow':
            new_input = np.expand_dims(np.expand_dims(input_, axis=2), axis=0)
        elif len(input_.shape) == 2 and K.backend() == 'theano':
            new_input = np.expand_dims(np.expand_dims(input_, axis=0), axis=0)
        else:
            new_input = np.expand_dims(input_, axis=0)

        predictValue = self.model.predict(new_input)
        newClass = np.argmax(np.ravel(predictValue))
        confident = np.amax(np.ravel(predictValue))
        return newClass, confident

# ------------------------------------------------------------------------------

    def load_weights(self):
        if K.backend() == 'tensorflow':
            self.model.load_weights('%s/mnist_tensorflow.h5' % self.DIRECTORY)
        else:
            weightFile = '%s/mnist.mat' % self.DIRECTORY
            modelFile = '%s/mnist.json' % self.DIRECTORY
            weights = sio.loadmat(weightFile)
            self.model = model_from_json(open(modelFile).read())
            for idx, lvl in [(1, 0), (2, 2), (3, 7), (4, 10)]:
                weight_1 = 2 * idx - 2
                weight_2 = 2 * idx - 1
                self.model.layers[lvl].set_weights([
                    weights['weights'][0, weight_1],
                    weights['weights'][0, weight_2].flatten()
                ])
