# Written by: 	Suren Gourapura
# Written on: 	June 19, 2018
# Purpose: 	To write a Convolutional Neural Network
# Source:	Following directions from: http://deeplearning.stanford.edu/wiki/index.php/Exercise:Convolution_and_Pooling
# Goal:		Test the Convolve and Pooling code


import numpy as np
from random import randint


# Calculate the Hypothesis for a1 -> a2
def hypoA12(W, b, dat):
	newhypo = sig(np.matmul(W, dat.T) +b)
	return newhypo.T  

def sig(x):
	return 1.0 / (1.0 + np.exp(-x))

def CheckConv(imgs, convImgs, W, b, ZCAmat, mpatch):
	dimrow = imgs.shape[1]-8+1	# 57
	dimcol = imgs.shape[2]-8+1	# 57
	for i in range(1000): #1000
		# randint(a,b) chooses random int x so that a <= x <= b
		featnum = randint(0, W.shape[0]-1)
		imgnum 	= randint(0, 7)			# Only test on the first 8
		imrow	= randint(0, dimrow-1)
		imcol	= randint(0, dimcol-1)
		# Select random 8x8x3 patches to test with
		testImg = imgs[imgnum, imrow:imrow+8, imcol:imcol+8, :]
		# Need to flatten the image, specifically in groups of color
		testImg = np.concatenate((testImg[:,:,0].flatten(),testImg[:,:,1].flatten(),testImg[:,:,2].flatten() ))
		# Subtract the mean, whiten, and forward propagate
		testImg -= np.ravel(mpatch)
		testImg = testImg.reshape(1,192).dot(ZCAmat)
		testFeat = hypoA12(W, b, testImg)	# 1 x 400
				# (400, 2k, 57, 57)
		if abs(convImgs[featnum,imgnum,imrow,imcol] - testFeat[0,featnum]) > 1e-9:
			print 'problem here at', i, imgnum
			print convImgs[featnum,imgnum,imrow,imcol] - testFeat[0,featnum]

	print 'Convolution check is ok'
	print ' '
	return


