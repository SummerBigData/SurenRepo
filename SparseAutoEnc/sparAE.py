# Written by: 	Suren Gourapura
# Written on: 	June 5, 2018
# Purpose: 	To write a Sparce Auto-Encoder following directions from: http://deeplearning.stanford.edu/wiki/index.php/Exercise:Sparse_Autoencoder
# Goal:		Python code to calculate W vals


import numpy as np
#from math import log
from scipy.optimize import minimize
import scipy.io
import time
import argparse
import matplotlib.pyplot as plt
#from scipy.optimize import check_grad
from random import randint

import randpicGen



#---------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES



parser = argparse.ArgumentParser()
parser.add_argument("m", help="Number of Datapoints", type=int)
#parser.add_argument("f1", help="Number of Features (pixels) in images", type=int)
#parser.add_argument("f2", help="Number of Features in hidden layer", type=int)
parser.add_argument("lamb", help="Lambda, the overfitting knob", type=float)
parser.add_argument("beta", help="Beta, sparsity knob", type=float)
#parser.add_argument("eps", help="Bounds for theta matrix randomization, [-eps, eps]", type=float)
parser.add_argument("tolexp", help="Exponent of tolerance of minimize function, good value 10e-4, so -4", type=int)
parser.add_argument("randData", help="Use fresh, random data or use the saved data file (true or false)", type=str)

g = parser.parse_args()

gStep = 0
g.eps = 0.12
g.f1 = 64
g.f2 = 25
g.rho = 0.05

saveStr = 'WArrs/m' + str(g.m)+ 'Tol'+str(g.tolexp)+'Lamb'+str(g.lamb)+'fone'+str(g.f1)+'ftwo'+str(g.f2)+'.out'


print 'You have chosen:', g
print 'Will be saved in:', saveStr
print ' '



#----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE


# Save the the
def saveW(vec):
	np.savetxt(saveStr, vec, delimiter=',')


# Generate random W matrices with a range [-eps, eps]
def randMat(x, y):
	theta = np.random.rand(x,y) 	# Makes a (x) x (y) random matrix of [0,1]
	return theta*2*g.eps - g.eps	# Make it range [-eps, eps]


# Linearize: Take 2 matrices, unroll them, and stitch them together into a vector
def Lin2(a, b):
	return np.concatenate((np.ravel(a), np.ravel(b)))
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


# Calculate the Hypothesis (for layer l to l+1)
def hypothesis(W, b, dat):
	oldhypo = np.matmul(W, dat.T) + b
	oldhypo = np.array(oldhypo, dtype=np.float128)	# Helps prevent overflow errors
	newhypo = 1.0/(1.0+np.exp(-oldhypo))	
	return np.array(newhypo.T, dtype=np.float64)


# Calculate the Hypothesis (layer 3) using just layer 1.
def ForwardProp(WAll, a1):
	W1, W2, b1, b2 = unLinWAll(WAll)
	# Calculate a2 (g.m x 25)
	a2 = hypothesis(W1, b1, a1)
	# Calculate and return the output from a2 and W2 (g.m x 64)
	a3 = hypothesis(W2, b2, a2)
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
	J = J + 0.5 * g.lamb * (np.sum(W1**2)+np.sum(W2**2))
	J = J + g.beta * np.sum(   g.rho*np.log(g.rho / phat) + (1-g.rho)*np.log((1-g.rho)/(1-phat))  )

	return J




#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE



# DATA PROCESSING
# To see how long the code runs for, we start a timestamp
totStart = time.time()

# Get data. Grab the saved data
picDat = np.genfromtxt('data/rand10kSAVED.out', dtype=float)

# If user wants fresh data, run randpicGen.py and rewrite picDat with this data
if g.randData == 'true':
	randpicGen.GenDat()
	picDat = np.genfromtxt('data/rand10k.out', dtype=float)

# Roll up data into matrix. Restrict it to [0,1]. Trim array to user defined size
dat = np.asarray(picDat.reshape(10000,64))/ 2.0 +0.5
dat = dat[0:g.m, :]

# Prepare the W matrices and b vectors and linearize them
W1 = randMat(g.f2, g.f1)
W2 = randMat(g.f1, g.f2)
b1 = randMat(g.f2, 1)
b2 = randMat(g.f1, 1)
WAll = Lin4(W1, W2, b1, b2) # 1D vector, probably length 13289


# CALCULATING IDEAL W MATRICES
# Check the cost of the initial W matrices
print 'Initial W JCost: ', RegJCost(WAll, dat) 

## Check the gradient function. ~1.0405573537e-05 for randomized thetas
## print check_grad(RegJCost, BackProp, thetaAll, xArr, yArr)

## Calculate the best theta values for a given j and store them. Usually tol=10e-4
#res = minimize(fun=RegJCost, x0= thetaAll, method='CG', tol=10**g.tolexp, jac=BackProp, args=(xArr, yArr))
#bestThetas = res.x

#print 'Final Theta JCost', RegJCost(bestThetas, xArr, yArr)

#saveTheta(bestThetas)

## Stop the timestamp and print out the total time
#totend = time.time()
#print'neuTrainerMNIST.py took ', totend - totStart, 'seconds to run'












# Stop the timestamp and print out the total time
totend = time.time()
print ' '
print'sparAE.py took ', totend - totStart, 'seconds to run'



