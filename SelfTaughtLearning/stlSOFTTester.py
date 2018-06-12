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


#---------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES


f1 = 784
f2 = 200
f3 = 10


#----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE


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

# Get data. Call the data by acccessing the function in dataPrep
dat, y = dataPrep.PrepData('04')
# Total Data size 30596. Using second half: length 15298
a1 = dat[len(y)/2:, :]
y = y[len(y)/2:]
#dat = Norm(dat)
#print np.amax(dat), np.amin(dat)

# Prepare the W matrices and b vectors and linearize them. Use the autoencoder W1 and b1, but NOT W2, b2
bestWAE = np.genfromtxt('WArrs/FullLamb10Btest/L10B0.5/m29404Tol-4Lamb10.0beta0.5.out', dtype=float)
W1, W2AE, b1, b2AE = unLinWAllAE(bestWAE)	# W1: 200 x 784, b1: 200 x 1
WA1 = LinW(W1, b1)	# 1D vector, probably length 157000

WA2 = np.genfromtxt('WArrs/FullLamb10Btest/L10B0.5/Tol-4Lamb0.0.out', dtype=float)

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
numPercent = np.array([0.0 for i in range(5)])
numNumbers = np.array([0 for i in range(5)])

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






## SHOW IMAGES
#hspaceAll = np.asarray([ [0 for i in range(116)] for j in range(10)])
#picAll = hspaceAll

#for i in range(10):
#	# Store the pictures
#	picA1 = np.reshape(np.ravel(a1[i]), (28,28))
#	picA2 = np.reshape(np.ravel(a2[i]), (20,10))
#	picA3 = np.reshape(np.ravel(a3[i]), (28,28))
##	print np.linalg.norm(a1[i*100])
##	print np.linalg.norm(a3[i*100])
#	# DISPLAY PICTURES
#	# To display a2 in revprop, a1, and a2 in forward prop, we design some spaces
#	hspace = np.zeros((28,10))
#	vspace = np.zeros((4, 20))

#	# We stitch the vertical spaces onto the pictures
#	picA2All = np.concatenate((vspace, picA2, vspace), axis = 0)
#	# We stitch the horizontal pictures together
#	picAlli = np.concatenate((hspace, picA1, hspace, picA2All, hspace, picA3, hspace), axis = 1)
#	# Finally, add this to the picAll
#	picAll = np.vstack((picAll, picAlli, hspaceAll))

## Display the pictures
#imgplot = plt.imshow(picAll, cmap="binary", interpolation='none') 
#plt.savefig('results/SoftRes/a123'+'L10B0.5Lamb0.0.png',transparent=False, format='png')
#plt.show()



## We also want a picture of the activations for each node in the hidden layer
#W1, b1 = unLinW1(W)
#W1Len = np.sum(W1**2)**(-0.5)
#X = W1 / W1Len			# (25 x 64)
#X = Norm(X)

#picX = np.zeros((25,8,8))
#for i in range(25):
#	picX[i] = np.reshape(np.ravel(X[i]), (8,8))

#hblack = np.asarray([ [1 for i in range(52)] for j in range(2)])
#vblack = np.asarray([ [1 for i in range(2)] for j in range(8)])

#picAll = hblack
#for i in range(5):
#	pici = np.concatenate((vblack, picX[5*i+0], vblack, picX[5*i+1], vblack, picX[5*i+2], vblack, picX[5*i+3], vblack, picX[5*i+4], vblack), axis = 1)
#	picAll = np.vstack((picAll, pici, hblack))

## Display the pictures
#imgplot = plt.imshow(picAll, cmap="binary", interpolation='none') 
#plt.savefig('results/aHL'+'Tol'+str(g.tolexp)+'Lamb'+str(g.lamb)+'rand'+g.randData+'.png',transparent=False, format='png')
#plt.show()



