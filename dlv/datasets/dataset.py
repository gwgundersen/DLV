import matplotlib
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------

class Dataset(object):
    """Abstract class representing DLV-supported dataset.
    """

    def __init__(self):
        pass

# ------------------------------------------------------------------------------

    def labels(self, index):
        raise NotImplementedError()

# ------------------------------------------------------------------------------

    def load(self):
        raise NotImplementedError()

# ------------------------------------------------------------------------------

    def save(self, layer, image, filename):
        """Save a given numpy.uint8 2D array of pixel data.
        """
        matplotlib.pyplot.switch_backend('agg')

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        imgplot = ax.imshow(image * 255, cmap=matplotlib.cm.Greys)
        imgplot.set_interpolation('nearest')
        ax.xaxis.set_ticks_position('top')
        ax.yaxis.set_ticks_position('left')
        plt.savefig(filename)

# ------------------------------------------------------------------------------

    def show(self, image):
        """Render a given numpy.uint8 2D array of pixel data.
        """
        matplotlib.pyplot.switch_backend('agg')

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        imgplot = ax.imshow(image, cmap=matplotlib.cm.Greys)
        imgplot.set_interpolation('nearest')
        ax.xaxis.set_ticks_position('top')
        ax.yaxis.set_ticks_position('left')
        plt.show()

