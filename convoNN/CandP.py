# Written by: 	Suren Gourapura
# Written on: 	June 18, 2018
# Purpose: 	To write a Convolutional Neural Network
# Source:	Following directions from: http://deeplearning.stanford.edu/wiki/index.php/Exercise:Convolution_and_Pooling
# Goal:		Use a trained autoencoder to convolve and pool features


import numpy as np
from math import floor
import time
import argparse
import dataPrep
import testCandP
from scipy.signal import convolve2d




#---------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES



parser = argparse.ArgumentParser()
parser.add_argument("m", help="Number of images, usually 2k", type=int)
parser.add_argument("CPrate", help="Rate at which we convolve and pool, usually 100", type=int)
g = parser.parse_args()
g.f1 = 192
g.f2 = 400

datStr = 'data/m100000Tol-4Lamb0.003beta5.0.out'
saveStr = 'convolvedData/m' + str(g.m) + 'CPRate' + str(g.CPrate) + '.out'
print 'You have chosen:', g
print ' '



#----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE



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
	# To see how long the code runs for, we start a timestamp
	totStart = time.time()
	print 'Running Convolution'

	# 400 x 2k x 57 x 57
	convImgs = np.zeros((WT.shape[0], imgs.shape[0], imgs.shape[1]-8+1, imgs.shape[2]-8+1))

	for i in range(imgs.shape[0]):			# For each image
		for f in range(WT.shape[0]):		# For each feature
			
			convPatch = np.zeros((imgs.shape[1]-8+1,imgs.shape[2]-8+1))	# 57 x 57

			for c in range(imgs.shape[3]):		# For each color
				convPatch += conv(imgs[i,:,:,c], WT[f,:,:,c])	#(64, 64) (8, 8) -> 57 x 57
			#convPatch /= 3.0 			# We want average of colors, not sum
			convPatch += bmean[f,0]			# Add the b - WT.xbar term
			convImgs[f,i,:,:]= sig(convPatch);	# passing the mean 57 x 57 patch from all channels

	# Stop time and print out
	totend = time.time()
	print "Convolution took ", totend - totStart, 'seconds to run'
	return convImgs


def Pool(convImgs, pooldim): # convImgs is (400, 2k, 57, 57)
	# To see how long the code runs for, we start a timestamp
	totStart = time.time()
	print 'Running Pool'

	poolImgs = np.zeros((convImgs.shape[0],convImgs.shape[1],pooldim,pooldim))
	poolstep = int(convImgs.shape[2]/pooldim)	# assuming square
	# loop over the 57 x 57, calculate where to store each convImg in poolImgs, and store it
	for i in range(convImgs.shape[2]):
		for j in range(convImgs.shape[2]):
			rowpos = int(floor(i/(poolstep+0.0)))
			colpos = int(floor(j/(poolstep+0.0)))
			poolImgs[:,:,rowpos,colpos] += convImgs[:,:,i,j]
	# Stop time and print out
	totend = time.time()
	print "Pooling took ", totend - totStart, 'seconds to run'
	return poolImgs/(0.0+poolstep**2)


#def SquarePatch(patch): # Expecting (1, 192) or 192 vec
#	patchrav = np.ravel(patch)
#	patchSq = np.zeros((8,8,3))
#	patchSq[:,:,0] = patchrav[:64].reshape(8, 8)
#	patchSq[:,:,1] = patchrav[64:128].reshape(8, 8)
#	patchSq[:,:,2] = patchrav[128:].reshape(8, 8)
#	return patchSq


# Unlinearize: Take a vector, break it into two vectors, and roll it back up
def unLinWAll(vec):	
	W1 = np.asarray([vec[0			: g.f2*g.f1]])
	#W2 = np.asarray([vec[g.f2*g.f1 		: g.f2*g.f1*2]])
	b1 = np.asarray([vec[g.f2*g.f1*2 	: g.f2*g.f1*2 + g.f2]])
	#b2 = np.asarray([vec[ g.f2*g.f1*2 + g.f2 : g.f2*g.f1*2 + g.f2 + g.f1]])
	#return W1.reshape(g.f2, g.f1) , W2.reshape(g.f1, g.f2), b1.reshape(g.f2, 1), b2.reshape(g.f1, 1)
	return W1.reshape(g.f2, g.f1), b1.reshape(g.f2, 1)



#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE



# DATA PROCESSING

# Get data. Call the data by acccessing the function in dataPrepColor
patches = dataPrep.GenDat()	# 100k x 64 x 3
imgs = dataPrep.GenSubTrain()	# 2k x 64 x 64 x 3
imgs = imgs[:g.m,:,:,:]

# Whiten data
wpatches, ZCAmat, patches, mpatch = dataPrep.SamzcaWhite(patches)
#(100k, 192) (192, 192) (100k, 192) (1, 192)

# Get the W and b arrays
bestWAll = np.genfromtxt(datStr, dtype=float)
W1, b1 = unLinWAll(bestWAll)	#(400, 192) (400, 1)


# CONVOLVE AND POOL
# Prepare for Convolution by calculating WT and bmean (b - WT.xbar term)
WT, bmean = ConvPrep(W1, ZCAmat, b1, mpatch)	# (400, 8, 8, 3) (400, 1)

# Initialize the data that will be sent back
poolImgs = np.zeros((W1.shape[0], g.m, 3, 3)) # Number can be 1, 3, 19, or 57. 3 is most reasonable

# Convolve and Pool the images. We do 100 at a time
for i in range(int( (imgs.shape[0]+0.0)/g.CPrate ) ):
	print "Batch #:",i+1
	convImgs = Convolve(imgs[i*g.CPrate : i*g.CPrate+g.CPrate], WT, bmean)		# (400, 100, 57, 57)
	poolImgs[:, i*g.CPrate : i*g.CPrate+g.CPrate, :,:] = Pool(convImgs, 3) # Number can be 1, 3, 19, or 57. 3 is most reasonable
	# (400, 100, 3, 3)
	print ' '

np.savetxt(saveStr, np.ravel(poolImgs), delimiter=',')



# Test Pooling
#test = np.arange(81).reshape((1,1,9,9))
#print test
#print Pool(test, 3)

# Test Convolve
# Check the Convolution
# testCandP.CheckConv(imgs, convImgs, W1, b1, ZCAmat, mpatch)



