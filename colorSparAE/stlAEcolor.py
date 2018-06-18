# Written by: 	Suren Gourapura
# Written on: 	June 7, 2018
# Purpose: 	To write a Self-Taught Learning Algorithim using MNIST dataset
# Source:	Following directions from: http://deeplearning.stanford.edu/wiki/index.php/Exercise:Self-Taught_Learning
# Goal:		Read 5-9 data and train an autoencoder


import numpy as np
#from math import log
from scipy.optimize import minimize
import scipy.io
import time
import argparse
import matplotlib.pyplot as plt
from scipy.optimize import check_grad
#from random import randint
import dataPrepColor


#---------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES



parser = argparse.ArgumentParser()
parser.add_argument("m", help="Number of Datapoints, up to 100k", type=int)
#parser.add_argument("f1", help="Number of Features (pixels) in images", type=int)
#parser.add_argument("f2", help="Number of Features in hidden layer", type=int)
parser.add_argument("lamb", help="Milli-lambda, the overfitting knob", type=float)
parser.add_argument("beta", help="deci-beta, sparsity knob", type=float)
#parser.add_argument("eps", help="Bounds for theta matrix randomization, [-eps, eps]", type=float)
parser.add_argument("tolexp", help="Exponent of tolerance of minimize function, good value 10e-4, so -4", type=int)
#parser.add_argument("oak", help="Is this code being run on oakley or on a higher python version?", type=str)

g = parser.parse_args()

#g.m = 0 # Will be adjusted later
gStep = 0
g.eps = 0.12
g.f1 = 192
g.f2 = 400 #400
g.rho = 0.035
#g.beta = 3
g.lamb /= 1.0
#g.beta /= 10.0

saveStr = 'WArrs/m' + str(g.m)+ 'Tol'+str(g.tolexp)+'Lamb'+str(g.lamb)+'beta'+str(g.beta)+'.out'


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


# Linearize: Take 4 matrices, unroll them, and stitch them together into a vector
def Lin4(a, b, c, d):
	return np.concatenate((np.ravel(a), np.ravel(b), np.ravel(c), np.ravel(d)))

# Unlinearize: Take a vector, break it into two vectors, and roll it back up
def unLinWAll(vec):	
	W1 = np.asarray([vec[0			: g.f2*g.f1]])
	W2 = np.asarray([vec[g.f2*g.f1 		: g.f2*g.f1*2]])
	b1 = np.asarray([vec[g.f2*g.f1*2 	: g.f2*g.f1*2 + g.f2]])
	b2 = np.asarray([vec[ g.f2*g.f1*2 + g.f2 : g.f2*g.f1*2 + g.f2 + g.f1]])
	return W1.reshape(g.f2, g.f1) , W2.reshape(g.f1, g.f2), b1.reshape(g.f2, 1), b2.reshape(g.f1, 1)


# Calculate the Hypothesis for a1 -> a2
def hypoA12(W, b, dat):
	oldhypo = np.matmul(W, dat.T) + b
	oldhypo = np.array(oldhypo, dtype=np.float128)	# Helps prevent overflow errors
	newhypo = 1.0/(1.0+np.exp(-oldhypo))	
	return np.array(newhypo.T, dtype=np.float64)

# Calculate the Hypothesis for a2 -> a3
def hypoA23(W, b, dat):
	oldhypo = np.matmul(W, dat.T) + b
	return oldhypo.T

# Calculate the Hypothesis (layer 3) using just layer 1.
def ForwardProp(WAll, a1):
	W1, W2, b1, b2 = unLinWAll(WAll)
	# Calculate a2 (g.m x 25)
	a2 = hypoA12(W1, b1, a1)
	# Calculate and return the output from a2 and W2 (g.m x 64)
	a3 = hypoA23(W2, b2, a2)
	return a2, a3

# Calculate the regularized Cost J(theta)
def RegJCost(WAll, a1):
	# Forward Propagate
	a2, a3 = ForwardProp(WAll, a1)
	# Seperate and reshape the Theta values
	W1, W2, b1, b2 = unLinWAll(WAll)
	# Calculate Sparsity contribution. Hehe, phat sounds like fat (stands for p hat)
	phat = (1.0 / g.m)*np.sum(a2, axis=0) # 25 len vector
	# Calculate J(W, b)
	J = (0.5/g.m)*np.sum((a3 - a1)**2)
	J = J + 0.5*g.lamb * (np.sum(W1**2)+np.sum(W2**2))
	J = J + g.beta * np.sum(   g.rho*np.log(g.rho / phat) + (1-g.rho)*np.log((1-g.rho)/(1-phat))  )
	return J

# Calculate the gradient of cost function for all values of W1, W2, b1, and b2
def BackProp(WAll, a1):
	# To keep track of how many times this code is called
	global gStep
	gStep += 1
	if gStep % 50 == 0:
		print 'Global Step: ', gStep, 'with JCost: ',  RegJCost(WAll, a1)
	if gStep % 200 == 0:
		print 'Saving Global Step : ', gStep
		saveW(WAll)
	# Seperate and reshape the W and b values
	W1, W2, b1, b2 = unLinWAll(WAll)
	# Forward Propagate
	a2, a3 = ForwardProp(WAll, a1)	# a2 (g.m x 25), a3 (g.m x 64)
	# Creating (Capital) Delta matrices
	DeltaW1 = np.zeros(W1.shape)			# (g.f2, g.f1)
	DeltaW2 = np.zeros(W2.shape)			# (g.f1, g.f2)
	Deltab1 = np.zeros(b1.shape)			# (g.f2, 1)
	Deltab2 = np.zeros(b2.shape)			# (g.f1, 1)

	# Calculate (Lowercase) deltas for each element in the dataset and add it's contributions to the Deltas
	#delta3 = np.multiply( -1*(a1 - a3), a3*(1-a3) )
	delta3 = -(a1 - a3)
	# Calculate Sparsity contribution to delta2
	phat = (1.0 / g.m)*np.sum(a2, axis=0)
	sparsity = g.beta * ( -g.rho/phat + (1-g.rho)/(1-phat)	)
	delta2 = np.multiply( np.matmul(delta3, W2) + sparsity.reshape(1, g.f2), a2*(1-a2) )

	DW1 = np.dot(delta2.T, a1) 	# (25, 64)
	DW2 = np.dot(delta3.T, a2)     	# (64, 25)
	Db1 = np.mean(delta2, axis = 0) # (25,) vector
	Db2 = np.mean(delta3, axis = 0) # (64,) vector

	return Lin4( (1.0/g.m)*DW1 + g.lamb*W1 , (1.0/g.m)*DW2 + g.lamb*W2 , Db1 , Db2 )

def Norm(mat):
	Min = np.amin(mat)
	Max = np.amax(mat)
	nMin = 0.00001
	nMax = 0.99999
	return ((mat - Min) / (Max - Min)) * (nMax - nMin) + nMin



#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE 



# DATA PROCESSING
# To see how long the code runs for, we start a timestamp
totStart = time.time()

# Get data. Call the data by acccessing the function in dataPrepColor
dat = dataPrepColor.GenDat()	# 100k x 64 x 3
dat = dat[:g.m, :, :]
whitenedDat, ZCAmat, dat = dataPrepColor.SamzcaWhite(dat)

# Another way, pull the matrix from the saved data
#ZCAmat = np.genfromtxt('data/m100.0kZCA.out', dtype=float).reshape(192,192)

# Reshape and normalize the data
a1 = whitenedDat.reshape(g.m, g.f1)
for i in range(g.m):
	a1[i] = Norm(a1[i])

#print np.amax(dat), np.amin(dat)



# Prepare the W matrices and b vectors and linearize them
W1 = randMat(g.f2, g.f1)
W2 = randMat(g.f1, g.f2)
b1 = randMat(g.f2, 1)
b2 = randMat(g.f1, 1)
WAll = Lin4(W1, W2, b1, b2) # 1D vector, probably length 3289



# CALCULATING IDEAL W MATRICES
# Check the cost of the initial W matrices
print 'Initial W JCost: ', RegJCost(WAll, a1) 

# Check the gradient. Go up and uncomment the import check_grad to use. ~6.38411247537693e-05 for 100 for randomized Ws and bs w/ g.f2=20
#print check_grad(RegJCost, BackProp, WAll, a1)

# Calculate the best theta values for a given j and store them. Usually tol=10e-4. usually 'CG'
## Since python 2.7.8 wants dat to be wrapped in another array, we use this
#if g.oak == 'true':
#	arg = np.asarray([a1])
#elif g.oak == 'false':
#	arg = a1

arg = a1
res = minimize(fun=RegJCost, x0= WAll, method='L-BFGS-B', tol=10**g.tolexp, jac=BackProp, args=(arg,) ) # options = {'disp':True}
bestWAll = res.x

print 'Final W JCost', RegJCost(bestWAll, a1)

saveW(bestWAll)

# Stop the timestamp and print out the total time
totend = time.time()
print ' '
print'sparAE.py took ', totend - totStart, 'seconds to run'



