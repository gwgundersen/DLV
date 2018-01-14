import numpy as np
import torch
from torch.utils.serialization import load_lua
from dlv.nn.nn import NN

# ------------------------------------------------------------------------------

class SvhnNN(NN):

    def __init__(self):
        self.define()
        print(self.model)

# ------------------------------------------------------------------------------

    def define(self):
        self.model = load_lua('dlv/nn/svhn/ln5_cpu.t7')

# ------------------------------------------------------------------------------

    def predict(self, x):
        batch    = torch.Tensor(1, 3, 32, 32)
        batch[0] = torch.Tensor(x)
        y = self.model.forward(batch)
        y = y.clone().numpy()
        class_ = np.argmax(np.ravel(y))
        confidence = np.amax(np.ravel(y))
        return class_, confidence
