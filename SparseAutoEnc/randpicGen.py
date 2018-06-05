# Written by: 	Suren Gourapura
# Written on: 	June 5, 2018
# Purpose: 	To write a Sparce Auto-Encoder following directions from: http://deeplearning.stanford.edu/wiki/index.php/Exercise:Sparse_Autoencoder
# Goal:		Generate 10k pictures


import numpy as np
import time
import matplotlib.pyplot as plt
from random import randint
import scipy.io


#---------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES



saveStr = 'data/rand10k.out'



#----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE


# Plot the entire 10 image dataset, 5 images in a row, two in a column
def PlotAll(dat):
	pic1 = dat[0]
	for i in range(4):
		pic1 = np.concatenate((pic1, dat[i+1]), axis = 1)

	pic2 = dat[5]
	for i in range(4):
		pic2 = np.concatenate((pic2, dat[i+4]), axis = 1)

	PlotImg(np.vstack((pic1, pic2)) )
#	imgplot = plt.imshow(picAll, cmap="binary", interpolation='none') 
#	plt.show()

def PlotImg(mat):
	imgplot = plt.imshow(mat, cmap="binary", interpolation='none') 
	plt.show()



def savePics(vec):
	np.savetxt(saveStr, vec, delimiter=',')



#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE

def GenDat():
	# To see how long the code runs for, we start a timestamp
	totStart = time.time()
	print 'Running randpicGen.py'
	print 'Will be saved in: ', saveStr


	# Get the data. It is 512 x 512 x 10, so we convert it to 10 x 512 x 512
	data = scipy.io.loadmat('data/IMAGES.mat')
	dat = np.zeros((10, 512, 512))
	for i in range(512):
		for j in range(512):
			for k in range(10):
				dat[k,i,j] = data['IMAGES'][i,j,k]
	
	# For plotting the data
	#PlotAll(dat)

	sampleDat = np.zeros((10000, 8, 8))
	for i in range(10000):
		wPic = randint(0, 9) # Pick one of the 10 images
		wRow = randint(0,504)# Pick a row for the first element
		wCol = randint(0,504)# Pick a column for the first element

#		# For plotting one of the images
#		imgplot = plt.imshow(dat[wPic,wRow:wRow+8,wCol:wCol+8], cmap="binary", interpolation='none') 
#		plt.show()

		sampleDat[i]=dat[wPic,wRow:wRow+8,wCol:wCol+8]
	

	savePics(np.ravel(sampleDat))

	# Stop the timestamp and print out the total time
	totend = time.time()
	print'randpicGen.py took ', totend - totStart, 'seconds to run'























