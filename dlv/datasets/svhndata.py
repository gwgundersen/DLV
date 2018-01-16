import numpy as np
from keras import backend as K
from torch.utils.serialization import load_lua
import matplotlib
import matplotlib.pyplot as plt

from dlv.datasets.dataset import Dataset


# ------------------------------------------------------------------------------

class SvhnData(Dataset):
    """A DLV-supported representation of the SVHN dataset.
    """

    name = 'svhn'

    def __init__(self):
        train = load_lua('dlv/datasets/svhn_raw/train_32x32_normalized.t7')
        test  = load_lua('dlv/datasets/svhn_raw/test_32x32_normalized.t7')
        self.X_train = train['data']
        self.Y_train = train['labels']
        self.X_test  = test['data']
        self.Y_test  = test['labels']

# ------------------------------------------------------------------------------

    def getTestImage(self, indx):
        """Return image.
        """
        image = self.X_test[indx:indx + 1]
        # Convert Torch tensor to NumPy ndarray.
        image = image.numpy()
        if K.backend() == 'tensorflow':
            return image[0]
        else:
            return np.squeeze(image)

# ------------------------------------------------------------------------------

    def labels(self, index):
        return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'][index]

# ------------------------------------------------------------------------------

    def show(self, image):
        """Render a given numpy.uint8 2D array of pixel data.
        """
        import pdb; pdb.set_trace()
        # matplotlib.pyplot.switch_backend('agg')
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # imgplot = ax.imshow(image, cmap=matplotlib.cm.Greys)
        # imgplot.set_interpolation('nearest')
        # ax.xaxis.set_ticks_position('top')
        # ax.yaxis.set_ticks_position('left')
        # plt.show()

    def save(self, layer, image, filename):
        """Save a given numpy.uint8 2D array of pixel data.
        """
        image = self.normalize(image)
        matplotlib.pyplot.switch_backend('agg')

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        img = (image * 255).T
        imgplot = ax.imshow(img)
        imgplot.set_interpolation('nearest')
        ax.xaxis.set_ticks_position('top')
        ax.yaxis.set_ticks_position('left')
        plt.savefig(filename)

    def normalize(self, image):
        min_ = image.min()
        max_ = image.max()
        return (image - min_) / (max_ - min_)