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
import dataPrep
from scipy.signal import convolve2d


parser = argparse.ArgumentParser()
#parser.add_argument("m", help="Number of Datapoints, usually 100000", type=int)
#parser.add_argument("f1", help="Number of Features (pixels) in images", type=int)
#parser.add_argument("f2", help="Number of Features in hidden layer", type=int)
#parser.add_argument("lamb", help="Lambda, the overfitting knob", type=float)
#parser.add_argument("beta", help="Beta, sparsity knob", type=float)
#parser.add_argument("eps", help="Bounds for theta matrix randomization, [-eps, eps]", type=float)
#parser.add_argument("tolexp", help="Exponent of tolerance of minimize function, good value 10e-4, so -4", type=int)

g = parser.parse_args()


g.f1 = 192
g.f2 = 400
g.m = 100000

saveStr = 'data/m1000Tol-4Lamb0.003beta5.0.out'

print 'You have chosen:', g
print ' '





def conv(img, mat):
	matRevTP = np.flipud(np.fliplr(mat))
	return convolve2d(img, matRevTP, mode='valid', boundary='fill', fillvalue=0)

def sig(x):
	return 1.0 / (1.0 + np.exp(-x))

def Convolve(imgs, W, b, ZCAmat, mpatch):
	
	convImg = np.zeros(( ))
	for i in range(imgs.shape[0]):			# For each image
		for j in range(W.shape[0]):		# For each feature
			for k in range(imgs.shape[3]):	# For each color
				WT = W[j].dot(ZCAmat)
				convImg = sig( WT.dot(imgs[i,:,:,k]) - WT.dot(mpatch) + b) 
	return convImg

# Unlinearize: Take a vector, break it into two vectors, and roll it back up
def unLinWAll(vec):	
	W1 = np.asarray([vec[0			: g.f2*g.f1]])
	#W2 = np.asarray([vec[g.f2*g.f1 		: g.f2*g.f1*2]])
	b1 = np.asarray([vec[g.f2*g.f1*2 	: g.f2*g.f1*2 + g.f2]])
	#b2 = np.asarray([vec[ g.f2*g.f1*2 + g.f2 : g.f2*g.f1*2 + g.f2 + g.f1]])
	#return W1.reshape(g.f2, g.f1) , W2.reshape(g.f1, g.f2), b1.reshape(g.f2, 1), b2.reshape(g.f1, 1)
	return W1.reshape(g.f2, g.f1), b1.reshape(g.f2, 1)




# DATA PROCESSING

# Get data. Call the data by acccessing the function in dataPrepColor
patches = dataPrep.GenDat()	# 100k x 64 x 3
imgs = dataPrep.GenSubTrain()	# 2000 x 64 x 64 x 3

# Whiten data
wpatches, ZCAmat, patches, mpatch = dataPrep.SamzcaWhite(patches)
#(100000, 192) (192, 192) (100000, 192) (1, 192)

# Reorganize the mean patch as 8 x 8 x 3
mpatch = np.ravel(mpatch)
mSqpatch = np.zeros((8,8,3))
mSqpatch[:,:,0] = mpatch[:64].reshape(8, 8)
mSqpatch[:,:,1] = mpatch[64:128].reshape(8, 8)
mSqpatch[:,:,2] = mpatch[128:].reshape(8, 8)

# Get the W and b arrays
bestWAll = np.genfromtxt(saveStr, dtype=float)
W1, b1 = unLinWAll(bestWAll)	#(400, 192) (400, 1)

print Convolve(imgs, W1, b1, ZCAmat, mSqpatch).shape





