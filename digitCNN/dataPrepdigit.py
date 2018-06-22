# Written by: 	Suren Gourapura
# Written on: 	June 12, 2018
# Purpose: 	To write a linear Sparce Auto-Encoder following directions from: http://deeplearning.stanford.edu/wiki/index.php/Exercise:Learning_color_features_with_Sparse_Autoencoders
# Goal:		Generate the dataset and send it to the main code


import numpy as np
import time
import scipy.io
import struct as st
import gzip
from random import randint, seed

#---------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES

saveStr = 'data/patches15m10kpart'
patchdim = 15

#----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE

# Save the WAll values
def save(vec, number):
	np.savetxt(saveStr+str(number)+'.out', np.ravel(vec), delimiter=',')

# Read the MNIST dataset
def read_idx(filename, n=None):
	with gzip.open(filename) as f:
		zero, dtype, dims = st.unpack('>HBB', f.read(4))
		shape = tuple(st.unpack('>I', f.read(4))[0] for d in range(dims))
		arr = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
		if not n is None:
			arr = arr[:n]
		return arr

#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE


def GenTrain():
	# To see how long the code runs for, we start a timestamp
	totStart = time.time()
	print 'Running dataPrep.py, GenTrain'

	# Get the data. It is 64 x 64 x 3 x 2000
	datx = read_idx('data/train-images-idx3-ubyte.gz', 60000)
	daty = read_idx('data/train-labels-idx1-ubyte.gz', 60000)

	# Get the data in matrix form
	datx = np.ravel(datx).reshape((60000, 784))

	# Stop the timestamp and print out the total time
	totend = time.time()
	print"dataPrep.py's GenTrain took ", totend - totStart, 'seconds to run'
	print ' '
	return datx, daty


def GenTest():
	# To see how long the code runs for, we start a timestamp
	totStart = time.time()
	print 'Running dataPrep.py, GenTest'

	# Get the data. It is 64 x 64 x 3 x 2000
	datx = read_idx('data/t10k-images-idx3-ubyte.gz', 10000)
	daty = read_idx('data/t10k-labels-idx1-ubyte.gz', 10000)

	# Get the data in matrix form
	datx = np.ravel(datx).reshape((10000, 784))

	# Stop the timestamp and print out the total time
	totend = time.time()
	print"dataPrep.py's GenTest took ", totend - totStart, 'seconds to run'
	print ' '
	return datx, daty


def GenPatches(dat):	# Gets training data as 60k x 784
	# To see how long the code runs for, we start a timestamp
	totStart = time.time()
	print 'Running dataPrepdigit.py, GenPatches'

	dat = dat.reshape((60000, 28, 28))

	patches = np.zeros((10000, patchdim, patchdim))
	seed(7)
	for i in range(10000):
		wPic = randint(0, 60000) 	# Pick one of the 10 images
		wRow = randint(0,28 -patchdim)	# Pick a row for the first element
		wCol = randint(0,28 -patchdim)	# Pick a column for the first element
		
		patches[i] = dat[wPic,wRow:wRow+patchdim,wCol:wCol+patchdim]

	numFiles = 4				# Save the data in 4 pieces
	fileLen = int(10000.0 / numFiles)
	for i in range(numFiles):	
		save( patches[i*fileLen : (i+1)*fileLen], i+1 )

	# Stop the timestamp and print out the total time
	totend = time.time()
	print"dataPrepdigit.py's GenPatches took ", totend - totStart, 'seconds to run'
	print ' '



# Create the patched data for training an autoencoder 
datx, daty = GenTrain()
GenPatches(datx)



