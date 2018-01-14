from dlv import Solver
from dlv.nn import SvhnNN
from dlv.datasets import SvhnData, MnistData

model   = SvhnNN()
dataset = SvhnData()

s = Solver(model, dataset)
s.verify()

