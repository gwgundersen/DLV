import numpy as np
from keras import backend as K
from torch.utils.serialization import load_lua
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchvision

from dlv.datasets.dataset import Dataset
from torchvision.datasets.svhn import SVHN as SvhnLoader

# ------------------------------------------------------------------------------

class SvhnData(Dataset):
    """A DLV-supported representation of the SVHN dataset.
    """

    name = 'svhn'

    def __init__(self):
        SUBDIR = '/Users/gwg/dlv/dlv/datasets/svhn_raw/'
        train = SvhnLoader(SUBDIR, split='train')
        test  = SvhnLoader(SUBDIR, split='test')
        self.X_train = train.data / 255.0
        self.Y_train = train.labels
        self.X_test  = test.data / 255.0
        self.Y_test  = test.labels

# ------------------------------------------------------------------------------

    def getTestImage(self, indx):
        """Return image.
        """
        image = self.X_test[indx:indx + 1]
        # Convert Torch tensor to NumPy ndarray.
        if K.backend() == 'tensorflow':
            return image[0]
        else:
            return np.squeeze(image)

# ------------------------------------------------------------------------------

    def labels(self, index):
        return [None, '1', '2', '3', '4', '5', '6', '7', '8', '9', '0'][index]

# ------------------------------------------------------------------------------

    def save(self, layer, image, filename):
        """Save a given numpy.uint8 2D array of pixel data.
        """
        torchvision.utils.save_image(torch.Tensor(image), filename)

        # import pdb;  pdb.set_trace()
        # image = self.normalize(image)
        # matplotlib.pyplot.switch_backend('agg')
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        #
        # img = (image * 255).T
        # imgplot = ax.imshow(img)
        # imgplot.set_interpolation('nearest')
        # ax.xaxis.set_ticks_position('top')
        # ax.yaxis.set_ticks_position('left')
        # plt.savefig(filename)

    def normalize(self, image):
        # min_ = image.min()
        # max_ = image.max()
        # return (image - min_) / (max_ - min_)
        image[image < 0] = 0
        image[image > 1] = 1
        return image
