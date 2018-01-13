from keras import backend as K
import numpy as np

# ------------------------------------------------------------------------------

class NN(object):

    def __init__(self):
        pass

# ------------------------------------------------------------------------------

    def define(self):
        raise NotImplementedError()

# ------------------------------------------------------------------------------

    def load_weights(self):
        raise NotImplementedError()

# ------------------------------------------------------------------------------

    def predict(self):
        raise NotImplementedError()

# ------------------------------------------------------------------------------

    def getLayerType(self, layer2Consider):
        if layer2Consider == -1:
            return 'Input'
        else:
            config = self.getConfig()
            # Get the type of the current layer.
            layerType = [lt for (l, lt) in config if layer2Consider == l]
            if len(layerType) > 0:
                layerType = layerType[0]
            else:
                print('cannot find the layerType')

            return layerType

# ------------------------------------------------------------------------------

    def getConfig(self):
        config = self.model.get_config()
        if 'layers' in config:
            config = config['layers']
        config = [getLayerName(dict_) for dict_ in config]
        config = zip(range(len(config)), config)
        return config

# ------------------------------------------------------------------------------

    def getActivationValue(self, layer, image):
        if len(image.shape) == 2:
            image = np.expand_dims(np.expand_dims(image, axis=0), axis=0)
        elif len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        activations = self.getActivations(layer, image)
        return np.squeeze(activations)

# ------------------------------------------------------------------------------

    def getActivations(self, layer, X_batch):
        get_activations_fn = K.function(
            [self.model.layers[0].input, K.learning_phase()],
            self.model.layers[layer].output
        )
        return get_activations_fn([X_batch,0])

# ------------------------------------------------------------------------------

    def getWeightVector(self, layer2Consider):
        weightVector = []
        biasVector = []

        for layer in self.model.layers:
            index = self.model.layers.index(layer)
            h = layer.get_weights()

            if len(h) > 0 and index in [0, 2] and index <= layer2Consider:
                # for convolutional layer
                ws = h[0]

                # number of filters in the previous layer
                m = len(ws)
                # number of features in the previous layer
                # every feature is represented as a matrix
                n = len(ws[0])

                for i in range(1, m + 1):
                    biasVector.append((index, i, h[1][i - 1]))

                for i in range(1, m + 1):
                    v = ws[i - 1]
                    for j in range(1, n + 1):
                        # (feature, filter, matrix)
                        weightVector.append(((index, j), (index, i), v[j - 1]))

            elif len(h) > 0 and index in [7, 10] and index <= layer2Consider:
                # for fully-connected layer
                ws = h[0]

                # number of nodes in the previous layer
                m = len(ws)
                # number of nodes in the current layer
                n = len(ws[0])

                for j in range(1, n + 1):
                    biasVector.append((index, j, h[1][j - 1]))

                for i in range(1, m + 1):
                    v = ws[i - 1]
                    for j in range(1, n + 1):
                        weightVector.append(((index - 1, i), (index, j), v[j - 1]))

        return weightVector, biasVector

# ------------------------------------------------------------------------------

def getLayerName(dict):
    className = dict.get('class_name')
    if className == 'Activation':
        return dict.get('config').get('activation')
    else:
        return className