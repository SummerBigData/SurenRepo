# Written by: 	Suren Gourapura
# Written on: 	June 12, 2018
# Purpose: 	To write a linear Sparce Auto-Encoder following directions from: http://deeplearning.stanford.edu/wiki/index.php/Exercise:Learning_color_features_with_Sparse_Autoencoders
# Goal:		Generate the dataset and send it to the main code


import numpy as np
import time
#import matplotlib.pyplot as plt
import scipy.io

#---------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES



#----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE


def PlotImg(mat):
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
	print 'Running dataPrepColor.py'

	# Get the data. It is 192 x 100k, so we transpose into 100k x 192
	data = scipy.io.loadmat('data/stlSampledPatches.mat')['patches'].T
	data = data.reshape((100000, 64, 3))

	# Stop the timestamp and print out the total time
	totend = time.time()
	print'dataPrepColor.py took ', totend - totStart, 'seconds to run'
	
	return data

GenDat()


