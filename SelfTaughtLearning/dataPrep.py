# Written by: 	Suren Gourapura
# Written on: 	June 7, 2018
# Purpose: 	To write a Self-Taught Learning Algorithim using MNIST dataset
# Source:	Following directions from: http://deeplearning.stanford.edu/wiki/index.php/Exercise:Self-Taught_Learning
# Goal:		Python code to format data into 0-4 and 5-9


import numpy as np
import scipy.io
import struct as st
import gzip

#---------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES

# Save names for the data and labels
save04N = 'data/60knum0-4.out'
save59N = 'data/60knum5-9.out'

save04L = 'data/60klab0-4.out'
save59L = 'data/60klab5-9.out'

#----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE


# Save the WAll values
def saveW(saveStr, vec):
	np.savetxt(saveStr, vec, delimiter=',')

# Read the MNIST dataset
def read_idx(filename, n=None):
	with gzip.open(filename) as f:
		zero, dtype, dims = st.unpack('>HBB', f.read(4))
		shape = tuple(st.unpack('>I', f.read(4))[0] for d in range(dims))
		arr = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
		if not n is None:
			arr = arr[:n]
		return arr

def col(matrix, i):
    return np.asarray([row[i] for row in matrix])




#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE



# DATA PROCESSING
def PrepData(string):
	# Obtain the data values and convert them from arrays to lists
	datx = read_idx('data/train-images-idx3-ubyte.gz', 60000)
	daty = read_idx('data/train-labels-idx1-ubyte.gz', 60000)

	# Get the data in matrix form
	datx = np.ravel(datx).reshape((60000, 784))

	# Stick the data and labels together for now
	dat = np.hstack((daty.reshape(60000, 1), datx))

	# Organize the data with respect to the labels

	ind = np.argsort( dat[:,0] ).astype(int)
	ordDat = np.zeros(dat.shape)
	for i in range(len(ind)):
		ordDat[i] = dat[ind[i]]
	#ordDat = dat[dat[:,0].argsort()]

	# Find the index of the last 4. For some reason, this is a 1 element array still, so we choose the only element [0]
	last4Ind = np.argwhere(col(ordDat, 0)==4)[-1][0]
	# Seperate the data
	dat04 = ordDat[0:last4Ind+1]
	dat59 = ordDat[last4Ind+1: ]

	# Reorder the data
	np.random.seed(7)	# Some random seed
	np.random.shuffle(dat04)
	np.random.shuffle(dat59)

	if string == '04':
		return dat04[:,1:]/255.0, col(dat04, 0)
	elif string == '59':
		return dat59[:,1:]/255.0, col(dat59, 0)
	else:
		print 'Error, input is not "04" or "59"'
#	# Unravel and seperate data
#	x04 = np.ravel(dat04[1:])
#	x59 = np.ravel()
#	y04 = np.ravel(col(dat04, 0))
#	y59 = np.ravel()

#	# Save the data
#	print "Saving seperated data with shapes:", x04.shape, x59.shape, y04.shape, y59.shape
#	saveW(save04N, x04)
#	saveW(save59N, x59)
#	saveW(save04L, y04)
#	saveW(save59L, y59)


