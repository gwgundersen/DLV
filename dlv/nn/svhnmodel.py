import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.serialization import load_lua
from dlv.nn.model import Model

# ------------------------------------------------------------------------------

class SvhnModel(Model):

    def __init__(self):
        self.define()
        self.loss = nn.CrossEntropyLoss()
        print(self.model)

# ------------------------------------------------------------------------------

    def define(self):
        self.model = load_lua('dlv/nn/svhn/ln5_cpu.t7')

# ------------------------------------------------------------------------------

    def predict(self, x):
        batch    = torch.Tensor(1, 3, 32, 32)
        batch[0] = torch.Tensor(x)
        y = self.model.forward(batch)

        prob_dist = F.softmax(Variable(y), dim=1)
        prob_dist = prob_dist.data.clone().numpy()

        class_ = np.argmax(np.ravel(prob_dist))

        # SVHN documentation:
        #
        #     10 classes, 1 for each digit. Digit '1' has label 1, '9' has label
        #     9 and '0' has label 10.
        class_ = class_ + 1

        confidence = np.amax(np.ravel(prob_dist))
        assert(0 <= confidence <= 1)
        assert(1 <= class_ <= 10)
        return class_, confidence
