# Written by: 	Suren Gourapura
# Written on: 	June 11, 2018
# Purpose: 	To write a Self-Taught Learning Algorithim using MNIST dataset
# Source:	Following directions from: http://deeplearning.stanford.edu/wiki/index.php/Exercise:Self-Taught_Learning
# Goal:		Test the stlSOFT.py outputs


# Import the modules
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import scipy.io
import dataPrep
# For reading MNIST testing data directly
import struct as st
import gzip

#---------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES


f1 = 784
f2 = 200
f3 = 10


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


# Save the WAll values
def saveW(vec):
	np.savetxt(saveStr, vec, delimiter=',')


# Generate random W matrices with a range [-eps, eps]
def randMat(x, y):
	theta = np.random.rand(x,y) 	# Makes a (x) x (y) random matrix of [0,1]
	return theta*2*g.eps - g.eps	# Make it range [-eps, eps]


# Linearize: Take 2 matrices, unroll them, and stitch them together into a vector
def LinW(a, b):
	return np.concatenate((np.ravel(a), np.ravel(b)))

# Unlinearize AutoEncoder data: Take a vector, break it into two vectors, and roll it back up
def unLinWAllAE(vec):	
	W1 = np.asarray([vec[0			: f2*f1]])
	W2 = np.asarray([vec[f2*f1 		: f2*f1*2]])
	b1 = np.asarray([vec[f2*f1*2 	: f2*f1*2 + f2]])
	b2 = np.asarray([vec[ f2*f1*2 + f2 : f2*f1*2 + f2 + f1]])
	return W1.reshape(f2, f1) , W2.reshape(f1, f2), b1.reshape(f2, 1), b2.reshape(f1, 1)

# Unlinearize SOFT data: Take a vector, break it into two vectors, and roll it back up
def unLinW1(vec):	
	W1 = np.asarray([vec[0		: f2*f1]])
	b1 = np.asarray([vec[f2*f1	:]])
	return W1.reshape(f2, f1) , b1.reshape(f2, 1)
def unLinW2(vec):	
	W2 = np.asarray([vec[0		: f3*f2]])
	b2 = np.asarray([vec[f3*f2	:]])
	return W2.reshape(f3, f2) , b2.reshape(f3, 1)


# Calculate the Hypothesis (for layer l to l+1)
def hypothesis(W, b, dat):
	Max = np.amax(np.matmul(W, dat.T) + b)
	numer = np.exp( np.matmul(W, dat.T) + b - Max )	# 200 x 15298 for W1, b1
	denom = np.asarray([np.sum(numer, axis=0)])
	return (numer/denom).T


# Calculate the Hypothesis (layer 3) using just layer 1.
def ForwardProp(WA1, WA2, a1):
	W1, b1 = unLinW1(WA1)
	W2, b2 = unLinW2(WA2)
	# Calculate a2 (g.m x 200)
	a2 = hypothesis(W1, b1, a1)
	# Calculate and return the output from a2 and W2 (g.m x 10)
	a3 = hypothesis(W2, b2, a2)
	return a2, a3


def Norm(mat):
	Min = np.amin(mat)
	Max = np.amax(mat)
	nMin = 0.00001
	nMax = 0.99999
	return ((mat - Min) / (Max - Min)) * (nMax - nMin) + nMin


def col(matrix, i):
    return np.asarray([row[i] for row in matrix])



#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE 


# DATA PROCESSING

## Get data. Call the data by acccessing the function in dataPrep
#dat, y = dataPrep.PrepData('04')
## Total Data size 30596. Using second half: length 15298
#a1 = dat[len(y)/2:, :]
#y = y[len(y)/2:]

# Obtain the data values and convert them from arrays to lists
datx = read_idx('data/t10k-images-idx3-ubyte.gz', 10000)
daty = read_idx('data/t10k-labels-idx1-ubyte.gz', 10000)
datx = np.ravel(datx).reshape((10000, f1))

a1 = datx[:, :]/255.0
y = daty[:]

#dat = Norm(dat)
#print np.amax(dat), np.amin(dat)

# Prepare the W matrices and b vectors and linearize them. Use the autoencoder W1 and b1, but NOT W2, b2
bestWAE = np.genfromtxt('WArrs/60k/L100B0.5/m60000Tol-4Lamb100.0beta0.5.out', dtype=float)
W1, W2AE, b1, b2AE = unLinWAllAE(bestWAE)	# W1: 200 x 784, b1: 200 x 1
WA1 = LinW(W1, b1)	# 1D vector, probably length 157000

WA2 = np.genfromtxt('WArrs/60k/L100B0.5/SoftM60000Tol-4Lamb1e-12.out', dtype=float)

## Generate the y matrix. # 15298 x 10
#ymat = GenYMat(y)

# FORWARD PROPAGATE AND CALCULATE BEST GUESSES
# Feed the best W and b vals into forward propagation
a2, a3 = ForwardProp(WA1, WA2, a1)



# CALCULATE PERCENT CORRECT
# First, we need to collapse the 15298 x 10 into a 15298 array, holding the best guess for each element
guessBest = np.asarray([0 for i in range(len(y))])

for j in range(len(y)):
	guessBest[j] = np.ravel(a3[j]).argmax()	# Record the index of the highest value


# Now we want to see what percent of each number the code got right
# Since the data isn't distributed evenly, we also record the frequency of each number in the dataset
numPercent = np.zeros((10))
numNumbers = np.zeros((10))

# Increment numPercent[i] for each number i it gets right.
# Also increment the numNumbers array for whichever datapoint it was
for i in range(len(y)):
	numNumbers[ int(y[i]) ] += 1
	if guessBest[i] == y[i]:
		numPercent[ guessBest[i] ] += 1.0

print 'Number of data points identified correctly per number:'
print np.array2string(numPercent, separator=',')
print ' ' 
print 'Number of total data points identified correctly:', sum(numPercent), 'out of a total', sum(numNumbers)
numPercent = numPercent / numNumbers
print ' ' 
print 'Percent correct per number:' 
print np.array2string(numPercent, separator=',')
print' '
print'Total percent correct:', np.mean(numPercent)
print' '





# CALCULATE THE IDEAL NUMBERS, AGAIN
W1, b1 = unLinW1(WA1)
W2, b2 = unLinW2(WA2)

W1Len = np.sum(W1**2)**(-0.5)
X1 = W1 / W1Len			
#X1 = Norm(X1)

W2Len = np.sum(W2**2)**(-0.5)
X2 = W2 / W2Len			
#X2 = Norm(X2)

print X1.shape, np.amin(X1), np.amax(X1)
print X2.shape, np.amin(X2), np.amax(X2)
print a2.shape, np.amin(a2), np.amax(a2)

numer = np.zeros((10,200,784))
weight = np.zeros((10))
ideal = np.zeros((10,28,28))
for i in range(10):
	numer[i] = np.multiply(X2[i].reshape(200,1), X1)	# 200 x 784
	weight[i] = np.sum(X2[i])
	ideal[i] = np.sum(numer[i], axis=0).reshape((28,28))/weight[i]
	ideal[i] = Norm(ideal[i])

#print ideal0.shape, np.amin(ideal0), np.amax(ideal0)
hbar = np.ones((5, 10*28+11*5))
vbar = np.ones((28, 5))
picAll = vbar
for i in range(10):
	picAll  = np.hstack(( picAll, ideal[i], vbar))

picAll = np.vstack(( hbar, picAll, hbar))
imgplot = plt.imshow(picAll, cmap="binary", interpolation='none') 
#plt.savefig('results/Activ' + str(g.m)+ 'Tol'+str(g.tolexp)+'Lamb'+str(g.lamb)+'beta'+str(g.beta)+'.png',transparent=False, format='png')
plt.show()





## Pic Facts
#hlNumb = 15 	# Show how many of the hidden layers
#numbNumb = 20	# Show how many numbers


#vblack = np.ones((5, (hlNumb+3)*s+(hlNumb+4)*5 ))	
#hblack = np.ones((s,5))
#picAll = vblack

#for i in range(numbNumb):
#	# Reorder the elements in a2. First, create an array of the indicies
#	a2isortInd = np.argsort(a2[i])
#	# Now, sort the 0th picture according to the indicies, and also reorder the array to largest-->smallest
#	a2isort = a2[i][a2isortInd][::-1]
#	picXisort = picX[a2isortInd][::-1]
#	
#	# We want to do a weighted average of activations using a2 as weights
#	weightpicXi = np.zeros(picXisort.shape)
#	for j in range(len(a2isort)):
#		weightpicXi[j] = picXisort[j]*a2isort[j]
#	weight = np.sum(a2isort)
#	avgi = np.sum(weightpicXi, axis=0)/weight
##	print avgi.shape, np.amin(avgi), np.amax(avgi)

#	
#	picAlli = np.hstack((hblack, a1[i].reshape(s,s), hblack ))
#	for j in range(hlNumb):
#		picAlli = np.hstack((picAlli, picXisort[j], hblack))
#	
#	picAlli = np.hstack((picAlli, avgi, hblack, Norm(avgi), hblack))

#	picAll = np.vstack((picAll, picAlli, vblack))
## Display the pictures
#imgplot = plt.imshow(picAll, cmap="binary", interpolation='none') 
#plt.savefig('results/Activ' + str(g.m)+ 'Tol'+str(g.tolexp)+'Lamb'+str(g.lamb)+'beta'+str(g.beta)+'.png',transparent=False, format='png')
#plt.show()

