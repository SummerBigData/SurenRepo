# Written by: 	Suren Gourapura
# Written on: 	June 12, 2018
# Purpose: 	To write a linear Sparce Auto-Encoder following directions from: http://deeplearning.stanford.edu/wiki/index.php/Exercise:Learning_color_features_with_Sparse_Autoencoders
# Goal:		Generate the dataset and send it to the main code


import numpy as np
import time

import scipy.io

#---------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES

datlen = 100000
pixlen = 64
collen = 3
eps = 0.1

#----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE


def PlotImg(mat):
	import matplotlib.pyplot as plt
	imgplot = plt.imshow(mat, cmap="binary", interpolation='none') 
	plt.show()

def savePics(vec):
	np.savetxt(saveStr, vec, delimiter=',')

def col(matrix, i):
	return np.asarray([row[i] for row in matrix])


#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE

def GenDat():
	# To see how long the code runs for, we start a timestamp
	totStart = time.time()
	print 'Running dataPrepColor.py, GenDat'

	# Get the data. It is 192 x 100k, so we transpose into 100k x 192
	data = scipy.io.loadmat('data/stlSampledPatches.mat')['patches'].T
	data = data.reshape((datlen, pixlen, collen))

	# Stop the timestamp and print out the total time
	totend = time.time()
	print"dataPrepColor.py's GenDat took ", totend - totStart, 'seconds to run'
	print ' '
	return data

def ZCAwhite(patches): # Should be 100k x 64 x 3
	# To see how long the code runs for, we start a timestamp
	totStart = time.time()
	print 'Running dataPrepColor.py, ZCAwhite'
	# First, calculate and subtract the mean pixel
	mean = np.mean(patches, axis=1)		# 100k x 3, the average pixel in each image
	meanMat = np.zeros(patches.shape)	# We need this to be 100k x 64 x 3, so make a new matrix
	for i in range(datlen):
		for j in range(pixlen):
			meanMat[i,j]=mean[i]
	# Subtract the mean pixel in each image from all of the images
	patches -= meanMat

	# Now, calculate sigma
	patches = patches.reshape((datlen, pixlen*collen))
	sigma = np.matmul(patches.T, patches)/100000
	
	# Calculating SVD
	U, S, V = np.linalg.svd(sigma)
	
	# Calculate the Whitening
	ZCAWhite = np.diag(1.0/np.sqrt( np.ravel(S) + eps))
	ZCAWhite = np.matmul( U, ZCAWhite)
	ZCAWhite = np.matmul( ZCAWhite, U.T )
	print ZCAWhite.shape
	patches = np.dot(patches, ZCAWhite)

	# Stop the timestamp and print out the total time
	totend = time.time()
	print"dataPrepColor.py's ZCAwhite took ", totend - totStart, 'seconds to run'
	print ' '
	return patches.reshape((datlen, pixlen, collen))

def zca_whitening_matrix(X):
	# To see how long the code runs for, we start a timestamp
	totStart = time.time()
	print 'Running dataPrepColor.py, ZCAwhite'

	X = X.reshape((datlen, pixlen*collen))
  	"""
	Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
	INPUT:  X: [M x N] matrix.
	Rows: Variables
	Columns: Observations
	OUTPUT: ZCAMatrix: [M x M] matrix
	"""
	# Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
   	sigma = np.cov(X, rowvar=True) # [M x M]
	print sigma.shape
	# Singular Value Decomposition. X = U * np.diag(S) * V
   	U,S,V = np.linalg.svd(sigma)
        # U: [M x M] eigenvectors of sigma.
        # S: [M x 1] eigenvalues of sigma.
        # V: [M x M] transpose of U
	# Whitening constant: prevents division by zero
   	epsilon = eps
	# ZCA Whitening matrix: U * Lambda * U'
   	ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)) # [M x M]

	# Stop the timestamp and print out the total time
	totend = time.time()
	print"dataPrepColor.py's ZCAwhite took ", totend - totStart, 'seconds to run'
	print ' '
	return patches.reshape((datlen, pixlen, collen))


	return ZCAMatrix, np.dot(ZCAMatrix, dat)

def zca_whitening(inputs):

	# First, calculate and subtract the mean pixel
	mean = np.mean(inputs, axis=1)		# 100k x 3, the average pixel in each image

	meanMat = np.zeros(inputs.shape)	# We need this to be 100k x 64 x 3, so make a new matrix
	for i in range(datlen):
		for j in range(pixlen):
			meanMat[i,j]=mean[i]
	# Subtract the mean pixel in each image from all of the images
	#inputs -= meanMat	# Screw using the mean, images look terrible with it

	#print np.amin(inputs), np.amax(inputs)
	inputs = inputs.reshape((datlen, pixlen*collen))


	sigma = np.dot(inputs.T, inputs)/datlen #Correlation matrix
	#print sigma.shape
	U,S,V = np.linalg.svd(sigma) #Singular Value Decomposition
	epsilon = eps	             #Whitening constant, it prevents division by zero
	ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(S + epsilon))), U.T)                     		#ZCA Whitening matrix
	#print ZCAMatrix.shape, inputs.shape
	return np.dot(inputs, ZCAMatrix), ZCAMatrix
	#return ZCAMatrix, inputs

#ZCA, inputs = zca_whitening(GenDat())
dat = GenDat()
#print dat.shape
#ZCA = np.zeros((dat.shape))
#for i in range(datlen):
#	ZCA[i] = zca_whitening(dat[i]).reshape(pixlen, collen)

#print ZCA.shape
ZCA, ZCAmat = zca_whitening(dat)
print ZCA.shape, ZCAmat.shape

vline = np.ones((8, 1, 3))
hline = np.ones((1, 8*2+3, 3))
picAll = hline

for i in range(5):
	pici = np.hstack((vline, dat[i].reshape(8,8,3), vline, ZCA[i].reshape(8,8,3), vline))
	picAll = np.vstack((picAll, pici, hline))

PlotImg(picAll)



