# Written by: 	Suren Gourapura
# Written on: 	June 19, 2018
# Purpose: 	To write a Convolutional Neural Network
# Source:	Following directions from: http://deeplearning.stanford.edu/wiki/index.php/Exercise:Convolution_and_Pooling
# Goal:		Use convolved and pooled features to train a Neural Network on images


import numpy as np
from scipy.optimize import minimize
import scipy.io
import time
import argparse
#import matplotlib.pyplot as plt
from scipy.optimize import check_grad
import dataPrep



#---------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES


parser = argparse.ArgumentParser()
parser.add_argument("m", help="Number of images, usually 2k", type=int)
parser.add_argument("CPrate", help="Rate at which we convolve and pool, usually 100", type=int)
#parser.add_argument("f1", help="Number of Features (pixels) in images", type=int)
#parser.add_argument("f2", help="Number of Features in hidden layer", type=int)
parser.add_argument("lamb", help="Lambda, usually 1e-4", type=float)
#parser.add_argument("beta", help="Beta, sparsity knob", type=float)
#parser.add_argument("eps", help="Bounds for theta matrix randomization, [-eps, eps]", type=float)
#parser.add_argument("tolexp", help="Exponent of tolerance of minimize function, good value 10e-4, so -4", type=int)
g = parser.parse_args()
g.step = 0
g.f1 = 3600
g.f2 = 36
g.f3 = 4
g.tolexp = -4
g.eps = 0.12

datStr = 'convolvedData/m' + str(g.m) + 'CPRate' + str(g.CPrate) + '.out'
saveStr = 'WArrs/m' + str(g.m) + 'lamb' + str(g.lamb) + '.out'
print 'You have chosen:', g
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
	W1 = np.asarray([vec[0				: g.f2*g.f1]])
	W2 = np.asarray([vec[g.f2*g.f1 			: g.f2*g.f1 + g.f3*g.f2]])
	b1 = np.asarray([vec[g.f2*g.f1+g.f3*g.f2	: g.f2*g.f1+g.f3*g.f2 + g.f2]])
	b2 = np.asarray([vec[g.f2*g.f1+g.f3*g.f2 + g.f2 : ]])
	return W1.reshape(g.f2, g.f1) , W2.reshape(g.f3, g.f2), b1.reshape(g.f2, 1), b2.reshape(g.f3, 1)


# Calculate the Hypothesis
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
def RegJCost(WAll, a1, y):
	# Forward Propagate
	a2, a3 = ForwardProp(WAll, a1)
	# Seperate and reshape the Theta values
	W1, W2, b1, b2 = unLinWAll(WAll)
	# Calculate J(W, b)

	J = (0.5/g.m)*np.sum( -1*np.multiply(y, np.log(a3)) - np.multiply(1-y, np.log(1 - a3)) )
	J += (0.5 * g.lamb)*( np.sum(W1**2) + np.sum(W2**2) )
	return J


# Calculate the gradient of cost function for all values of W1, W2, b1, and b2
def BackProp(WAll, a1, y):
	# To keep track of how many times this code is called
	g.step += 1
	if g.step % 50 == 0:
		print 'Global Step: ', g.step, 'with JCost: ',  RegJCost(WAll, a1)
	if g.step % 200 == 0:
		print 'Saving Global Step : ', g.step
		saveW(WAll)
	# Seperate and reshape the W and b values
	W1, W2, b1, b2 = unLinWAll(WAll)
	# Forward Propagate
	a2, a3 = ForwardProp(WAll, a1)	# a2 (g.m x g.f2), a3 (g.m x g.f3)

	# Creating (Capital) Delta matrices
	DeltaW1 = np.zeros(W1.shape)			# (g.f2, g.f1)
	DeltaW2 = np.zeros(W2.shape)			# (g.f3, g.f2)
	Deltab1 = np.zeros(b1.shape)			# (g.f2, 1)
	Deltab2 = np.zeros(b2.shape)			# (g.f3, 1)

	# Calculate (Lowercase) deltas for each element in the dataset and add it's contributions to the Deltas
	delta3 = np.multiply( -1*(y - a3), a3*(1-a3) )
	delta2 = np.multiply( np.matmul(delta3, W2), a2*(1-a2) )

	DW1 = np.dot(delta2.T, a1) 	# (g.f2, g.f1)
	DW2 = np.dot(delta3.T, a2)     	# (g.f3, g.f2)
	Db1 = np.mean(delta2, axis = 0) # (g.f2,) vector
	Db2 = np.mean(delta3, axis = 0) # (g.f3) vector

	return Lin4( (1.0/g.m)*DW1 + g.lamb*W1 , (1.0/g.m)*DW2 + g.lamb*W2 , Db1 , Db2 )


# I am trying to keep the features with their 3 x 3 matrix here, 
# so the first image's first 9 values (a1[0,:8]) is the linearized 3x3 matrix for the first feature
def dat2D(cpdat):
	a1 = np.zeros((cpdat.shape[1], 400*3*3))
	for i in range(cpdat.shape[1]):
		# Create a temporary matrix holding linearized data for each image and feature
		temp = np.zeros((cpdat.shape[0], 9))
		for j in range(cpdat.shape[0]):
			temp[j,:] = np.ravel(cpdat[j,i,:,:])
		a1[i] = np.ravel(temp)
	return a1


#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE



# To see how long the code runs for, we start a timestamp
totStart = time.time()




# DATA PROCESSING
# Get the convolved and pooled images
cpdat = np.genfromtxt(datStr, dtype=float)
cpdat = cpdat.reshape((400, g.m, 3, 3))
# We need this in 2D form. g.m x 3600  (3 x 3 x 400 = 3600)
a1 = dat2D(cpdat)

# Get the y labels. Labeled in set [1, 4]
y = scipy.io.loadmat('data/stlTrainSubset.mat')['trainLabels']#.reshape(1,2000)
#y = y[:,:g.m]
y = y[:g.m,:]

# INITIALIZING W MATRICES
# Prepare the W matrices and b vectors and linearize them
W1 = randMat(g.f2, g.f1)
W2 = randMat(g.f3, g.f2)
b1 = randMat(g.f2, 1)
b2 = randMat(g.f3, 1)
WAll = Lin4(W1, W2, b1, b2) # 1D vector, probably length 3289


# CALCULATING IDEAL W MATRICES
# Check the cost of the initial W matrices
print 'Initial W JCost: ', RegJCost(WAll, a1, y) ,BackProp(WAll, a1, y)

# Check the gradient. Go up and uncomment the import check_grad to use. ~6.38411247537693e-05 for 100 for randomized Ws and bs w/ g.f2=20
print check_grad(RegJCost, BackProp, WAll, a1, y)


#arg = a1
#res = minimize(fun=RegJCost, x0= WAll, method='L-BFGS-B', tol=10**g.tolexp, jac=BackProp, args=(arg,) ) # options = {'disp':True}
#bestWAll = res.x

#print 'Final W JCost', RegJCost(bestWAll, a1)

#saveW(bestWAll)

## Stop the timestamp and print out the total time
#totend = time.time()
#print ' '
#print'sparAE.py took ', totend - totStart, 'seconds to run'


