from dlv import Solver
from dlv.nn import SvhnModel
from dlv.datasets import SvhnData, MnistData

model   = SvhnModel()
dataset = SvhnData()

s = Solver(model, dataset)
s.verify()

