# Written by: 	Suren Gourapura
# Written on: 	June 11, 2018
# Purpose: 	To write a Self-Taught Learning Algorithim using MNIST dataset
# Source:	Following directions from: http://deeplearning.stanford.edu/wiki/index.php/Exercise:Self-Taught_Learning
# Goal:		Combine with AutoEncoder using softmax regression to learn 0-4


import numpy as np
#from math import log
from scipy.optimize import minimize
import scipy.io
import time
import argparse
import matplotlib.pyplot as plt
#from scipy.optimize import check_grad
#from random import randint
import dataPrep


#---------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES



parser = argparse.ArgumentParser()
parser.add_argument("m", help="Number of Datapoints, usually 29404", type=int)
#parser.add_argument("f1", help="Number of Features (pixels) in images", type=int)
#parser.add_argument("f2", help="Number of Features in hidden layer", type=int)
parser.add_argument("lamb", help="Millilambda, the overfitting knob", type=float)
#parser.add_argument("beta", help="Tens of beta, sparsity knob", type=float)
#parser.add_argument("eps", help="Bounds for theta matrix randomization, [-eps, eps]", type=float)
parser.add_argument("tolexp", help="Exponent of tolerance of minimize function, good value 10e-4, so -4", type=int)
#parser.add_argument("oak", help="Is this code being run on oakley or on a higher python version?", type=str)

g = parser.parse_args()

#g.m = 0 # Will be adjusted later
gStep = 0
g.eps = 0.12
g.f1 = 784
g.f2 = 200
g.f3 = 10
#g.rho = 0.05
#g.beta = 3
g.lamb /= 1000.0


saveStr = 'WArrs/60k/L10B0.5/SoftM'+str(g.m)+'Tol'+str(g.tolexp)+'Lamb'+str(g.lamb)+'.out'


print 'You have chosen:', g
print 'Will be saved in:', saveStr
print ' '



#----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE


# Save the WAll values
def saveW(vec):
	np.savetxt(saveStr, vec, delimiter=',')


# Generate random W matrices with a range [-eps, eps]
def randMat(x, y):
	theta = np.random.rand(x,y) 	# Makes a (x) x (y) random matrix of [0,1]
	return theta*2*g.eps - g.eps	# Make it range [-eps, eps]


## Linearize: Take 4 matrices, unroll them, and stitch them together into a vector
#def LinWAll(a, b, c, d):
#	return np.concatenate((np.ravel(a), np.ravel(b), np.ravel(c), np.ravel(d)))
# Linearize: Take 2 matrices, unroll them, and stitch them together into a vector
def LinW(a, b):
	return np.concatenate((np.ravel(a), np.ravel(b)))

# Unlinearize AutoEncoder data: Take a vector, break it into two vectors, and roll it back up
def unLinWAllAE(vec):	
	W1 = np.asarray([vec[0			: g.f2*g.f1]])
	W2 = np.asarray([vec[g.f2*g.f1 		: g.f2*g.f1*2]])
	b1 = np.asarray([vec[g.f2*g.f1*2 	: g.f2*g.f1*2 + g.f2]])
	b2 = np.asarray([vec[ g.f2*g.f1*2 + g.f2 : g.f2*g.f1*2 + g.f2 + g.f1]])
	return W1.reshape(g.f2, g.f1) , W2.reshape(g.f1, g.f2), b1.reshape(g.f2, 1), b2.reshape(g.f1, 1)

# Unlinearize SOFT data: Take a vector, break it into two vectors, and roll it back up
def unLinW1(vec):	
	W1 = np.asarray([vec[0		: g.f2*g.f1]])
	b1 = np.asarray([vec[g.f2*g.f1	:]])
	return W1.reshape(g.f2, g.f1) , b1.reshape(g.f2, 1)
def unLinW2(vec):	
	W2 = np.asarray([vec[0		: g.f3*g.f2]])
	b2 = np.asarray([vec[g.f3*g.f2	:]])
	return W2.reshape(g.f3, g.f2) , b2.reshape(g.f3, 1)


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

# Calculate the regularized Cost J(theta)
def RegJCost(WA2, WA1, a1, ymat):
	# Forward Propagate
	a2, a3 = ForwardProp(WA1, WA2, a1)
	# Seperate and reshape the Theta values
	W2, b2 = unLinW2(WA2)
	# Calculate J(W, b). ymat and a3 are the same shape: 15298 x 10
	return (-1.0 / len(y))*np.sum( np.multiply(np.log(a3), ymat)  ) + g.lamb*0.5*np.sum(W2**2)


def BackProp(WA2, WA1, a1, ymat):
	# To keep track of how many times this code is called
	global gStep
	gStep += 1
	if gStep % 50 == 0:
		print 'Global Step: ', gStep, 'with JCost: ',  RegJCost(WA2, WA1, a1, ymat)
	if gStep % 200 == 0:
		print 'Saving Global Step : ', gStep
		saveW(WA2)

	# Forward Propagate
	a2, a3 = ForwardProp(WA1, WA2, a1)	# a2 (g.m x 200), a3 (g.m x 10)
	# Seperate and reshape the W and b values
	W2, b2 = unLinW2(WA2)
	
	# Now, to get backprop to work, I had to remake the theta matrices we had previously. Sandwich b2 onto W2
	WAll2 = np.hstack((b2, W2))
	# Attach a column of 1's onto a2
	left = np.array([[1] for i in range(len(col(ymat, 0))) ])
	a2ones = np.hstack((left, a2))
	# Calculate the derivative for both W2 and b2 at the same time
	DeltaWAll2 = (-1.0 / len(y))*np.matmul((ymat - a3).T, a2ones) + g.lamb*WAll2		# (g.f3, g.f2)
	# Seperate these back into W2 and b2 and linearize it
	return LinW(DeltaWAll2[:,1:], DeltaWAll2[:,:1])


def Norm(mat):
	Min = np.amin(mat)
	Max = np.amax(mat)
	nMin = 0.00001
	nMax = 0.99999
	return ((mat - Min) / (Max - Min)) * (nMax - nMin) + nMin

# Generate the y-matrix. This is called only once, so I use loops
def GenYMat(yvals):
	yvals = np.ravel(yvals)
	yArr = np.zeros((len(yvals), 10))
	for i in range(len(yvals)):
		for j in range(10):
			if yvals[i] == j:
				yArr[i][j] = 1
	return yArr

def col(matrix, i):
    return np.asarray([row[i] for row in matrix])



#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE 



# DATA PROCESSING
# To see how long the code runs for, we start a timestamp
totStart = time.time()

# Get data. Call the data by acccessing the function in dataPrep
dat, y = dataPrep.PrepData('09')
# Total Data size 30596. Using first half: length 15298
dat = dat[:g.m, :]	# len(y)/2 For 04, 59 testing
y = y[:g.m]
#dat = Norm(dat)
#print np.amax(dat), np.amin(dat)


# Prepare the W matrices and b vectors and linearize them. Use the autoencoder W1 and b1, but NOT W2, b2
bestWAll = np.genfromtxt('WArrs/60k/L10B0.5/m60000Tol-4Lamb10.0beta0.5.out', dtype=float)
W1, W2AE, b1, b2AE = unLinWAllAE(bestWAll)	# W1: 200 x 784, b1: 200 x 1
W2 = randMat(g.f3, g.f2)			# 10 x 200
b2 = randMat(g.f3, 1)				# 10 x 1
WA1 = LinW(W1, b1)	# 1D vector, probably length 157000
WA2 = LinW(W2, b2)	# 1D vector, probably length 2010

# Generate the y matrix. # 15298 x 10
ymat = GenYMat(y)



# CALCULATING IDEAL W MATRICES
# Check the cost of the initial W matrices
print 'Initial W JCost: ', RegJCost(WA2, WA1, dat, ymat) 
# Check the gradient. Go up and uncomment the import check_grad to use. ~2.378977939526638e-05 for m=98 for randomized Ws and bs
#print check_grad(RegJCost, BackProp, WA2, WA1, dat, ymat)

# Calculate the best theta values for a given j and store them. Usually tol=10e-4. usually 'CG'
## Since python 2.7.8 wants dat to be wrapped in another array, we use this
#if g.oak == 'true':
#	arg = np.asarray([dat])
#elif g.oak == 'false':
#	arg = dat

res = minimize(fun=RegJCost, x0= WA2, method='L-BFGS-B', tol=10**g.tolexp, jac=BackProp, args=(WA1, dat, ymat) ) # options = {'disp':True}
bestWA2 = res.x

print 'Final W JCost', RegJCost(bestWA2, WA1, dat, ymat) 

saveW(bestWA2)

# Stop the timestamp and print out the total time
totend = time.time()
print ' '
print'sparAE.py took ', totend - totStart, 'seconds to run'



