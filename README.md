# DLV

NB: This software is currently under active development. Please feel free to contact the main developer, Xiaowei Huang, by email: xiaowei.huang@cs.ox.ac.uk.

## Getting Started

Please refer to the original papers for more details about the software:

- [Feature-Guided Black-Box Safety Testing of Deep Neural Networks](docs/DLV_TheoryPaper.pdf)
- [Safety Verification of Deep Neural Networks](docs/DLV_MCTS_TwoPlayer.pdf)

## Installation

### Anaconda dependency manager

Before continuing, please install [Anaconda](https://docs.anaconda.com/anaconda/install/), a package manager, if needed. Currently, only Python 2.7 is supported. To create a Python 2.X specific Anaconda environment named `dlv`, do:

```bash
conda create --name dlv python=2.7 pip
```

Then activate the `dlv` environment:

```bash
source activate dlv
```

### Install dependencies
           
```bash
conda install opencv=2.4.8 numpy scikit-image cvxopt
```

The software currently does not work well with Keras 2.X because of image dimension ordering problems. Please use a previous 1.X version:

```bash
pip install keras==1.2.2 theano==0.9.0 pySMT z3 stopit tensorflow
```

The `z3` pacakge needs to be properly installed. You can follow the instruction: https://github.com/Z3Prover/z3, or run the following commands (tested on Mac OS): 

```bash
git clone https://github.com/Z3Prover/z3.git
cd z3
python scripts/mk_make.py --python
cd build
make
sudo make install
```
           
### Check the backend of Keras

The backend of Keras needs to be changed by editing the `~/.keras/keras.json` file: 

```json
"backend": "theano",
"image_dim_ordering": "th"
```

### Download dataset and network paramters

If you want to train a network for GTSRB, Please download the dataset file X.h5 file from https://www.dropbox.com/s/2brjdjghhnmw6i7/X.h5?dl=0 to networks/ directory. For details on download networks and datasets for imageNet, please refer to the document. 


## Usage

Use the following command to call the program: 

```bash
python DLV.py
```

Please use the file `configuration.py` to set the parameters for the system to run.