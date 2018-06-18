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

saveStr = 'data/m'+str(datlen/1000.0)+'kZCA.out'

#----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE

# Save the WAll values
def save(vec):
	np.savetxt(saveStr, vec, delimiter=',')

def PlotImg(mat):
	import matplotlib.pyplot as plt
	imgplot = plt.imshow(mat, cmap="binary", interpolation='none') 
	plt.show()

def col(matrix, i):
	return np.asarray([row[i] for row in matrix])

def Norm(mat):
	Min = np.amin(mat)
	Max = np.amax(mat)
	nMin = 0
	nMax = 1
	return ((mat - Min) / (Max - Min)) * (nMax - nMin) + nMin

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


def zcaWhite(inputs):	# To see how long the code runs for, we start a timestamp
	totStart = time.time()
	print 'Running dataPrepColor.py, ZCAwhite'
	# Reshape the data into 100k x 192
	datl = inputs.shape[0]
	pixl = inputs.shape[1]
	coll = inputs.shape[2]
	inputs = inputs.reshape((datl, pixl*coll))

	sigma = np.dot(inputs.T, inputs)/datl #Correlation matrix
	U,S,V = np.linalg.svd(sigma) #Singular Value Decomposition
        #eps is the whitening constant, it prevents division by zero
	ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(S + eps))), U.T)                     

	# Stop the timestamp and print out the total time
	totend = time.time()
	print"dataPrepColor.py's ZCAwhite took ", totend - totStart, 'seconds to run'
	print ' '

	return np.dot(inputs, ZCAMatrix), ZCAMatrix	# Return the whitened image and the matrix

def SamzcaWhite(X):
	# Reshape the data into 100k x 192
	datl = X.shape[0]
	pixl = X.shape[1]
	coll = X.shape[2]
	X = X.reshape((datl, pixl*coll))

	mu = np.mean(X, axis=0).reshape(1, X.shape[1])
	X -= np.tile(mu, (X.shape[0], 1))
	Sigma = np.dot(X.T, X)/X.shape[0]
	U, S, _ = np.linalg.svd(Sigma)
	S = S.reshape(len(S), 1)
	epsilon = 0.1
	l_mat = np.diag(1.0/np.sqrt(S.flatten() + epsilon))
	ZCA_mat = U.dot(l_mat).dot(U.T)
	return X.dot(ZCA_mat), ZCA_mat, X


#dat = GenDat()

## Reshape and normalize the data
##for i in range(dat.shape[0]):
##	dat[i] = Norm(dat[i])

#print np.amin(dat), np.amax(dat)
#print np.amin(dat[1]), np.amax(dat[1])

#ZCA, ZCAmat = zcaWhite(dat) # 100k x 192 and 192 x 192
##save(np.ravel(ZCAmat))

#ZCAsam, ZCAmatsam = SamzcaWhite(dat)

## Reshape and normalize the data
#for i in range(dat.shape[0]):
#	ZCA[i] = Norm(ZCA[i])
#	ZCAsam[i] = Norm(ZCAsam[i])

#print np.amin(dat[1]), np.amax(dat[1])
#print np.amin(ZCA[1]), np.amax(ZCA[1])
#print np.amin(ZCAsam[1]), np.amax(ZCAsam[1])
#ZCA = Norm(ZCA)


#vline = np.ones((8, 1, 3))
#hline = np.ones((1, 8*3+4, 3))
#picAll = hline

#for i in range(10):
#	pici = np.hstack((vline, dat[i].reshape(8,8,3), vline, ZCA[i].reshape(8,8,3), vline,ZCAsam[i].reshape(8,8,3), vline))
#	picAll = np.vstack((picAll, pici, hline))

#PlotImg(picAll)



