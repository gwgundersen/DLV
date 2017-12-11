#!/usr/bin/env python

"""
Define paramters
author: Xiaowei Huang
"""
import numpy as np
import os

import dlv.networks.mnist_network as NN_mnist
import dlv.networks.cifar10_network as NN_cifar10
import dlv.networks.imageNet_network as NN_imageNet
import dlv.networks.twoDcurve_network as NN_twoDcurve
import dlv.networks.gtsrb_network as NN_gtsrb

from dlv.networks import mnist
from dlv.networks import cifar10
from dlv.networks import imageNet
from dlv.networks import twoDcurve
from dlv.networks import gtsrb

def network_parameters(dataset): 

    
#######################################################
#
#  boundOfPixelValue for the elements of input 
#
#######################################################

    if dataset in ["mnist","gtsrb", "cifar10","imageNet"] : 
        boundOfPixelValue = [0,1]
    elif dataset == "twoDcurve": 
        boundOfPixelValue = [0, 2 * np.pi]
    else: 
        boundOfPixelValue = [0,0]

#######################################################
#
#  some auxiliary parameters that are used in several files
#  they can be seen as global parameters for an execution
#
#######################################################
    
# which dataset to analyse
    if dataset == "mnist": 
        NN = NN_mnist
        dataBasics = mnist
        directory_model_string = makedirectory("dlv/networks/mnist")
        directory_statistics_string = makedirectory("dlv/data/mnist_statistics")
        directory_pic_string = makedirectory("dlv/data/mnist_pic")
        
# ce: the region definition for layer 0, i.e., e_0
        span = 255/float(255)
        numSpan = 1
        errorBounds = {}
        errorBounds[-1] = 1.0
        
    elif dataset == "gtsrb": 
        NN = NN_gtsrb
        dataBasics = gtsrb
        directory_model_string = makedirectory("dlv/networks/gtsrb")
        directory_statistics_string = makedirectory("dlv/data/gtsrb_statistics")
        directory_pic_string = makedirectory("dlv/data/gtsrb_pic")
        
# ce: the region definition for layer 0, i.e., e_0
        span = 255/float(255)
        numSpan = 1
        errorBounds = {}
        errorBounds[-1] = 1.0
    
    elif dataset == "twoDcurve": 
        NN = NN_twoDcurve
        dataBasics = twoDcurve
        directory_model_string = makedirectory("dlv/networks/twoDcurve")
        directory_statistics_string = makedirectory("dlv/data/twoDcurve_statistics")
        directory_pic_string = makedirectory("dlv/data/twoDcurve_pic")

# ce: the region definition for layer 0, i.e., e_0
        span = 255/float(255) 
        numSpan = 1
        errorBounds = {}
        errorBounds[-1] = 1.0

    elif dataset == "cifar10": 
        NN = NN_cifar10
        dataBasics = cifar10
        directory_model_string = makedirectory("dlv/networks/cifar10")
        directory_statistics_string = makedirectory("dlv/data/cifar10_statistics")
        directory_pic_string = makedirectory("dlv/data/cifar10_pic")
 
# ce: the region definition for layer 0, i.e., e_0
        span = 255/float(255)
        numSpan = 1
        errorBounds = {}
        errorBounds[-1] = 1.0
                
    elif dataset == "imageNet": 
        NN = NN_imageNet
        dataBasics = imageNet
        directory_model_string = makedirectory("dlv/networks/imageNet")
        directory_statistics_string = makedirectory("dlv/data/imageNet_statistics")
        directory_pic_string = makedirectory("dlv/data/imageNet_pic")

# ce: the region definition for layer 0, i.e., e_0
        span = 125
        numSpan = 1
        errorBounds = {}
        errorBounds[-1] = 125
            
#######################################################
#
#  size of the filter used in convolutional layers
#
#######################################################
    
    filterSize = 3 
    return (span,numSpan,errorBounds,boundOfPixelValue,NN,dataBasics,directory_model_string,directory_statistics_string,directory_pic_string,filterSize)

def makedirectory(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    return directory_name