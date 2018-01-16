from dlv import Solver
from dlv.nn import SvhnModel, MnistModel
from dlv.datasets import SvhnData, MnistData

model   = SvhnModel()
dataset = SvhnData()

s = Solver(model, dataset)
s.verify(out_dir='out', idx=100)
