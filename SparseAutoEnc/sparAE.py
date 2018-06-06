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
#from random import randint
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
g.rho = 0.01

saveStr = 'WArrs/m' + str(g.m)+ 'Tol'+str(g.tolexp)+'Lamb'+str(g.lamb)+'fone'+str(g.f1)+'ftwo'+str(g.f2)+'rand'+g.randData+'.out'


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


## Linearize: Take 2 matrices, unroll them, and stitch them together into a vector
#def Lin2(a, b):
#	return np.concatenate((np.ravel(a), np.ravel(b)))
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
	J = J + (0.5/g.m)*g.lamb * (np.sum(W1**2)+np.sum(W2**2))
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

	delta3 = np.multiply( -1*(a1 - a3), a3*(1-a3) )
	# Calculate Sparsity contribution to delta2
	phat = (1.0 / g.m)*np.sum(a2, axis=0)
	sparsity = g.beta * ( -g.rho/phat + (1-g.rho)/(1-phat)	)
	delta2 = np.multiply( np.matmul(delta3, W2) + sparsity.reshape(1, g.f2), a2*(1-a2) )

	DW1 = np.dot(delta2.T, a1) 	# (25, 64)
	DW2 = np.dot(delta3.T, a2)     	# (64, 25)
	Db1 = np.mean(delta2, axis = 0) # (25,) vector
	Db2 = np.mean(delta3, axis = 0) # (64,) vector

#	for t in range(g.m):
#		delta3 = -1*np.multiply( (a1[t] - a3[t]), a3[t]*(1-a3[t]) )			# 64 vec
#		# Calculate Sparsity contribution to delta2
#		phat = (1.0 / g.m)*np.sum(a2, axis=0)						# 25 vec
#		sparsity = g.beta * ( -g.rho/phat + (1-g.rho)/(1-phat)	)			# 25 vec
#		delta2 = np.multiply( np.matmul(W2.T, delta3) + sparsity, a2[t]*(1-a2[t]) )	# 25 vec

#		DeltaW1 = DeltaW1 + np.outer(delta2, a1[t])		# (25 x 64)
#		DeltaW2 = DeltaW2 + np.outer(delta3, a2[t])		# (64 x 25)	
#		Deltab1 = Deltab1 + delta2.reshape((g.f2, 1))		# (25 x 1)
#		Deltab2 = Deltab2 + delta3.reshape((g.f1, 1))		# (64 x 1)
	
#	DerW1 = (1.0/g.m)*DeltaW1 + g.lamb*W1
#	DerW2 = (1.0/g.m)*DeltaW2 + g.lamb*W2
#	Derb1 = (1.0/g.m)*Deltab1
#	Derb2 = (1.0/g.m)*Deltab2
	return Lin4( (1.0/g.m)*(DW1 + g.lamb*W1) , (1.0/g.m)*(DW2 + g.lamb*W2) , Db1 , Db2 )

def Norm(mat):
	Min = np.amin(mat)
	Max = np.amax(mat)
	nMin = 0
	nMax = 1
	return ((mat - Min) / (Max - Min)) * (nMax - nMin) + nMin




#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE



# DATA PROCESSING
# To see how long the code runs for, we start a timestamp
totStart = time.time()

# Get data. Grab the saved data
picDat = np.genfromtxt('data/rand10kSAVE.out', dtype=float)

# If user wants fresh data, run randpicGen.py and rewrite picDat with this data
if g.randData == 'true':
	randpicGen.GenDat()
	picDat = np.genfromtxt('data/rand10k.out', dtype=float)

# Roll up data into matrix. Restrict it to [0,1]. Trim array to user defined size
dat = np.asarray(picDat.reshape(10000,64))
dat = dat[0:g.m, :]
# Normalize each image
for i in range(g.m):
	dat[i] = Norm(dat[i])

# Prepare the W matrices and b vectors and linearize them
W1 = randMat(g.f2, g.f1)
W2 = randMat(g.f1, g.f2)
b1 = randMat(g.f2, 1)
b2 = randMat(g.f1, 1)
WAll = Lin4(W1, W2, b1, b2) # 1D vector, probably length 3289

# CALCULATING IDEAL W MATRICES
# Check the cost of the initial W matrices
print 'Initial W JCost: ', RegJCost(WAll, dat) 

# Check the gradient. Go up and uncomment the import check_grad to use. ~1.84242805087e-05 for 100 for randomized Ws and bs
#print check_grad(RegJCost, BackProp, WAll, dat)

# Calculate the best theta values for a given j and store them. Usually tol=10e-4
res = minimize(fun=RegJCost, x0= WAll, method='CG', tol=10**g.tolexp, jac=BackProp, args=(dat))
bestWAll = res.x

print 'Final W JCost', RegJCost(bestWAll, dat)

saveW(bestWAll)

# Stop the timestamp and print out the total time
totend = time.time()
print ' '
print'sparAE.py took ', totend - totStart, 'seconds to run'



