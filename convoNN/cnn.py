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
from random import randint


parser = argparse.ArgumentParser()
parser.add_argument("m", help="Number of images, usually 2k", type=int)
#parser.add_argument("f1", help="Number of Features (pixels) in images", type=int)
#parser.add_argument("f2", help="Number of Features in hidden layer", type=int)
#parser.add_argument("lamb", help="Lambda, the overfitting knob", type=float)
#parser.add_argument("beta", help="Beta, sparsity knob", type=float)
#parser.add_argument("eps", help="Bounds for theta matrix randomization, [-eps, eps]", type=float)
#parser.add_argument("tolexp", help="Exponent of tolerance of minimize function, good value 10e-4, so -4", type=int)

g = parser.parse_args()


g.f1 = 192
g.f2 = 400

saveStr = 'data/m100000Tol-4Lamb0.003beta5.0.out'

print 'You have chosen:', g
print ' '




# The Convolve2d function flips the matrix before computing, so preliminarily flip it first
# Also, fill some of the parameters to make it easy to use in Convolve()
def conv(img, mat):
	matRevTP = np.flipud(np.fliplr(mat))
	return convolve2d(img, matRevTP, mode='valid', boundary='fill', fillvalue=0)


def ConvPrep(W, ZCAmat, b, mpatch):
	# Reorganize the W.T as a 400 x 8 x 8 x 3
	WT2d = W.dot(ZCAmat)	# (400, 192)
	WT = np.zeros((WT2d.shape[0], 8, 8, 3))
	for i in range(WT2d.shape[0]):
		WT[i,:,:,0] = WT2d[i,:64].reshape(8, 8)
		WT[i,:,:,1] = WT2d[i,64:128].reshape(8, 8)
		WT[i,:,:,2] = WT2d[i,128:].reshape(8, 8)

	# Calculate the added component bmean (b - WT.xbar term) 400 x 1
	bmean = b - WT2d.dot(mpatch.T)
	return WT, bmean


def sig(x):
	return 1.0 / (1.0 + np.exp(-x))

def Convolve(imgs, WT, bmean):
	# 400 x 2k x 57 x 57
	convImgs = np.zeros((WT.shape[0], imgs.shape[0], imgs.shape[1]-8+1, imgs.shape[2]-8+1))

	for i in range(imgs.shape[0]):			# For each image
		for f in range(WT.shape[0]):		# For each feature
			
			convPatch = np.zeros((imgs.shape[1]-8+1,imgs.shape[2]-8+1))	# 57 x 57

			for c in range(imgs.shape[3]):		# For each color
				convPatch += conv(imgs[i,:,:,c], WT[f,:,:,c])	#(64, 64) (8, 8) -> 57 x 57

			convPatch /= 3.0 			# We want average of colors, not sum
			convPatch += np.ravel(bmean)[f]		# Add the b - WT.xbar term
			
			convImgs[f,i,:,:]= sig(convPatch); # passing the mean 57 x 57 patch from all channels	
	return convImgs

#def SquarePatch(patch): # Expecting (1, 192) or 192 vec
#	patchrav = np.ravel(patch)
#	patchSq = np.zeros((8,8,3))
#	patchSq[:,:,0] = patchrav[:64].reshape(8, 8)
#	patchSq[:,:,1] = patchrav[64:128].reshape(8, 8)
#	patchSq[:,:,2] = patchrav[128:].reshape(8, 8)
#	return patchSq

# Calculate the Hypothesis for a1 -> a2
def hypoA12(W, b, dat):
	newhypo = sig(np.matmul(W, dat.T) +b)
	return newhypo.T  

def CheckConv(imgs, convImgs, W, b, ZCAmat, mpatch):
	dimrow = imgs.shape[1]-8+1	# 57
	dimcol = imgs.shape[2]-8+1	# 57
	

	for i in range(10): #1000
		# randint(a,b) chooses random int x so that a <= x <= b
		featnum = randint(0, W.shape[0]-1)
		imgnum 	= randint(0, 7)			# Only test on the first 8
		imrow	= randint(0, dimrow-1)
		imcol	= randint(0, dimcol-1)
		# Select random 8x8x3 patches to test with
		testImg = np.ravel( imgs[imgnum, imrow:imrow+8, imcol:imcol+8, :] )
		testImg -= np.ravel(mpatch)
		testImg = testImg.reshape(1,192).dot(ZCAmat)
		testFeat = hypoA12(W, b, testImg)	# 1 x 400
		
				# (400, 2k, 57, 57)
		if abs(convImgs[featnum,imgnum,imrow,imcol] - testFeat[0,featnum]) > 1e-9:
			print 'problem here at', i, imgnum
			print convImgs[featnum,imgnum,imrow,imcol] - testFeat[0,featnum]
			print ' '

	return 'ok'


def test_convolution(X, X_conv, W, b, Z, mu):
	# For sam's code
	from scipy.special import expit
    # Test the convolve function by picking 1000 random patches from the input data,
    # preprocessing it using Z and mu, and feeding it through the SAE (using W and b)
    #
    # If the result is close to the convolved patch, we're good

	patch_dim = int(np.sqrt(W.shape[1]/3.0)) # 8
	conv_dim = X.shape[1] - patch_dim + 1 # 57

	for i in range(100):
		feat_no = np.random.randint(0, W.shape[0])
		img_no = np.random.randint(0, X.shape[0])
		img_row = np.random.randint(0, conv_dim)
		img_col = np.random.randint(0, conv_dim)

		patch_x = (img_col, img_col+patch_dim)
		patch_y = (img_row, img_row+patch_dim)

		# Obtain a 8x8x3 patch and flatten it to length 192
		patch = X[img_no, patch_y[0]:patch_y[1], patch_x[0]:patch_x[1]]
		patch = np.concatenate((patch[:,:,0].flatten(), patch[:,:,1].flatten(), patch[:,:,2].flatten())).reshape(-1, 1)
		#patch = patch.reshape(-1, 1)

		# Preprocess the patch
		patch -= mu.reshape(192,1)
		patch = Z.dot(patch) 

		# Feed the patch through the autoencoder weights
		# now sae_patch.shape = (400 192) . (192 1) = (400 1)
		sae_feat = expit(W.dot(patch) + b)

		# Compare it to the convolved patch
		conv_feat = X_conv[:,img_no,img_row,img_col]
		#print conv_feat.reshape(20, 20)
		err = abs(sae_feat[feat_no, 0] - conv_feat[feat_no])
		"""
		import matplotlib.pyplot as plt
		img = np.zeros((20, 42))
		img[:,:20] = sae_feat.reshape(20, 20)
		img[:,22:] = conv_feat.reshape(20, 20)
		plt.imshow(img, cmap='gray')
		plt.show()
		"""
		if i == 5:
			exit()
		print err



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
imgs = dataPrep.GenSubTrain()	# 2k x 64 x 64 x 3
imgs = imgs[:g.m,:,:,:]

# Whiten data
wpatches, ZCAmat, patches, mpatch = dataPrep.SamzcaWhite(patches)
#(100k, 192) (192, 192) (100k, 192) (1, 192)

# Get the W and b arrays
bestWAll = np.genfromtxt(saveStr, dtype=float)
W1, b1 = unLinWAll(bestWAll)	#(400, 192) (400, 1)



# CONVOLVE
# Prepare for Convolution by calculating WT and bmean (b - WT.xbar term)
WT, bmean = ConvPrep(W1, ZCAmat, b1, mpatch)
# (400, 8, 8, 3) (400, 1)

convImgs = Convolve(imgs, WT, bmean)
# (400, 2k, 57, 57)

print CheckConv(imgs, convImgs, W1, b1, ZCAmat, mpatch)
print test_convolution(imgs, convImgs, W1, b1, ZCAmat, mpatch)




