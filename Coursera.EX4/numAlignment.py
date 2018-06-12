# Written by: 	Suren Gourapura
# Written on: 	June 12, 2018
# Purpose: 	To solve exercise 4 on Multi-class Classification and Neural Networks in Coursera
# Goal:		Align numbers to make it easier for the neural network to read them

# Import the modules
import numpy as np
import scipy.io
import time
import struct as st
import gzip
#import matplotlib.pyplot as plt
# For normImg
from numpy.polynomial import polynomial as P
from scipy.ndimage import rotate
from math import pi, atan
# This will be fun
import argparse

#---------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES

parser = argparse.ArgumentParser()
parser.add_argument("n", help="Number of Datapoints", type=int)
#parser.add_argument("f1", help="Number of Features (pixels) in images", type=int)
#parser.add_argument("f2", help="Number of Features in hidden layer", type=int)
#parser.add_argument("lamb", help="Lambda, the overfitting knob", type=float)
#parser.add_argument("eps", help="Bounds for theta matrix randomization, [-eps, eps]", type=float)
#parser.add_argument("tolexp", help="Exponent of tolerance of minimize function, good value 10e-4, so -4", type=int)
#parser.add_argument("normimg", help="Choose whether or not to straighten the images", type=str)
g = parser.parse_args()



g.f1 = 784


print 'You have chosen:', g
print ' '


#----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE


# Read the MNIST dataset
def read_idx(filename, n=None):
	with gzip.open(filename) as f:
		zero, dtype, dims = st.unpack('>HBB', f.read(4))
		shape = tuple(st.unpack('>I', f.read(4))[0] for d in range(dims))
		arr = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
		if not n is None:
			arr = arr[:n]
		return arr

def shiftImg(datx):
	# Get the size of the picture matrices
	s = int(np.sqrt(g.f1))
	messedMat = np.zeros(datx.shape)
	for i in range(g.n):
		# Convert it back to a matrix
		mat = np.reshape(datx[i], (s,s))

		# Mess with data:
		mat = np.delete(mat, 0, 0)			# Removes the first (top) row
		mat = np.vstack((mat, np.zeros((1, s)) ))	# Adds a bottom row of zeros

		messedMat[i] = np.ravel(mat)
	return messedMat





def normImg(datx):
	print "Normalizing data"
	# Get the size of the picture matrices
	s = int(np.sqrt(g.f1))
	# Create an array of numbers [1,2,3,4,...] to use for average position computation
	index = np.zeros((s))
	for i in range(s):
		index[i] = i + 1
	# Initialize the final matrix
	alignmat = np.zeros(datx.shape)

	# NUMBER STRAIGHTENING
	for i in range(g.n):
		# Convert it back to a matrix
		mat = np.reshape(datx[i], (s,s))

		# Create vectors to store the mean of the picture horizontally and vertically
		hmean = np.mean(mat, axis=1)
		vmean = np.mean(mat, axis=0)
		# Now we find the centers of each of these vectors to find the center of the number
		hcenter = sum(hmean*index)/ (sum(hmean)+0.0) - 1	# This is the center of axis 0
		vcenter = sum(vmean*index)/ (sum(vmean)+0.0) - 1	# This is the center of axis 1
		hcenter = int(round(hcenter, 0))
		vcenter = int(round(vcenter, 0))
		print 'before', hcenter, vcenter
		
		if hcenter < 14:
			shift = 14 - hcenter
			for j in range(shift):
				mat = np.delete(mat, s-1, 0)			# Delete bottom row of image
				mat = np.vstack((np.zeros((1, s)), mat ))	# Add a row of zeros on top

		if hcenter > 14:
			shift = hcenter - 14
			for j in range(shift):
				mat = np.delete(mat, 0, 0)			# Delete top row of image
				np.vstack((mat, np.zeros((1, s)) ))		# Add a row of zeros on bottom

		if vcenter < 14:
			shift = 14 - vcenter
			for j in range(shift):
				mat = np.delete(mat, s-1, 1)			# Delete right column of image
				mat = np.hstack((np.zeros((s,1)), mat ))	# Add a column of zeros on left
		
		if vcenter > 14:
			shift = vcenter - 14
			for j in range(shift):
				mat = np.delete(mat, 0, 1)			# Delete left column of image
				mat = np.hstack((mat, np.zeros((s,1)) ))	# Add a column of zeros on right
		

		# Create vectors to store the mean of the picture horizontally and vertically
		hmean = np.mean(mat, axis=1)
		vmean = np.mean(mat, axis=0)
		# Now we find the centers of each of these vectors to find the center of the number
		hcenter = sum(hmean*index)/ (sum(hmean)+0.0) - 1	# This is the center of axis 0
		vcenter = sum(vmean*index)/ (sum(vmean)+0.0) - 1	# This is the center of axis 1
		hcenter = int(round(hcenter, 0))
		vcenter = int(round(vcenter, 0))
		print 'after',  hcenter, vcenter

		alignmat[i] = np.ravel(mat)

#	# NUMBER ALIGNMENT
#	# Calculate the rotated matrix for all data points

#	for i in range(g.n):
#		# Convert it back to a matrix
#		mat = np.reshape(alignmat[i], (s,s))
#		hcenter = np.zeros((s))
#		# We need the horizontal centers for each row
#		for j in range(s):
#			# Handle the zero case seperately, due to divide by zero. The value here doesn't matter, since the weight will kill it
#			if sum(mat[j]) == 0:
#				hcenter[j] = -1
#			# Calculate and store the center of each column
#			else:
#				hcenter[j] = sum(mat[j]*index)/ (sum(mat[j])+0.0)
#		# We don't want to include the zero cases, so form a weights matrix to record them
#		weights = np.zeros((s))
#		for j in range(s):
#			if hcenter[j] < 0:
#				weights[j] = 0
#			else:
#				weights[j] = 1
##		print hcenter
#		# Calculate the line of best fit for all of the horizontal centers
#		c = P.polyfit(index,hcenter,1,full=False, w=weights)
#		# Here's some tools to visualize the process
##		print c[0], c[1], atan(c[1])*180.0/pi
##		bestfit = c[0] + c[1]*index
##		plt.plot(hcenter,'green', bestfit, 'red')
##		plt.show()
#		# Rotate, unravel, and record the matrix
#	
#		alignmat[i] = np.ravel(rotate(mat, -1*atan(c[1])*180.0/pi, reshape=False)).reshape(1, g.f1)

	return alignmat



#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE


# PREPARING DATA
# Obtain the data values and convert them from arrays to lists
datx = read_idx('data/train-images-idx3-ubyte.gz', g.n)
daty = read_idx('data/train-labels-idx1-ubyte.gz', g.n)
	
datx = np.ravel(datx).reshape((g.n, g.f1))


datx = shiftImg(datx)

datx = normImg(datx)





