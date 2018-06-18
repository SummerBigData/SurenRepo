# Written by: 	Suren Gourapura
# Written on: 	June 18, 2018
# Purpose: 	To write a Convolutional Neural Network
# Source:	Following directions from: http://deeplearning.stanford.edu/wiki/index.php/Exercise:Convolution_and_Pooling
# Goal:		Use a trained autoencoder and identify objects in images


import numpy as np
#from math import log
from scipy.optimize import minimize
import scipy.io
import time
import argparse
#import matplotlib.pyplot as plt
from scipy.optimize import check_grad
#from random import randint
#import dataPrepColor
from scipy.signal import convolve2d

def conv(img, mat):
	matRevTP = np.flipud(np.fliplr(mat))
	return convolve2d(img, matRevTP, mode='valid', boundary='fill', fillvalue=0)
