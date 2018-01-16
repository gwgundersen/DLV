import argparse

from dlv import Solver
from dlv.nn import SvhnModel
from dlv.datasets import SvhnData


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--download',  type=bool, default=False)
    parser.add_argument('--out_dir',   type=str,  default='out')
    parser.add_argument('--image_idx', type=int,  default=200)

    args = parser.parse_args()
    model   = SvhnModel()
    dataset = SvhnData(download=args.download)

    s = Solver(model, dataset)
    s.verify(out_dir=args.out_dir, idx=args.image_idx)
