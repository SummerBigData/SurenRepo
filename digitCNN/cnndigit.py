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
import dataPrepdigit



#---------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES


parser = argparse.ArgumentParser()
parser.add_argument("m", help="Number of images, usually 2k", type=int)
#parser.add_argument("CPrate", help="Rate at which we convolve and pool, usually 100", type=int)
#parser.add_argument("f1", help="Number of Features (pixels) in images", type=int)
parser.add_argument("f2", help="Number of Features in hidden layer", type=int)
parser.add_argument("lamb", help="Lambda, usually 1e-4", type=float)
#parser.add_argument("beta", help="Beta, sparsity knob", type=float)
#parser.add_argument("eps", help="Bounds for theta matrix randomization, [-eps, eps]", type=float)
#parser.add_argument("tolexp", help="Exponent of tolerance of minimize function, good value 10e-4, so -4", type=int)
g = parser.parse_args()
g.step = 0

#g.f2 = 36
g.f3 = 10
g.tolexp = -4
g.eps = 0.12
g.numfiles = 40
g.pooldim = 7
g.f1 = 100*g.pooldim**2

datStr = 'convolvedData/pooldim7trainingm60000patches15.out'
saveStr = 'WArrs/m' + str(g.m) + 'HL' +str(g.f2)+ 'lamb' + str(g.lamb) + '.out'
print 'You have chosen:', g
print ' '



#----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE


# Save the WAll values
def saveW(vec):
	np.savetxt(saveStr, vec, delimiter=',')


# Generate random W matrices with a range [-eps, eps]
def randMat(x, y):
	np.random.seed(7)
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


# Calculate the softmax Hypothesis (for layer 2 to 3)
def softHypo(W, b, a):
	Max = np.amax(np.matmul(W, a.T) + b)	# To not blow up the np.exp
	numer = np.exp( np.matmul(W, a.T) + b - Max )	
	denom = np.asarray([np.sum(numer, axis=0)])
	return (numer/denom).T
# Calculate the logistic Hypothesis (for layer 1 to 2)
def logHypo(W, b, a):
	oldhypo = np.matmul(W, a.T) + b
	#oldhypo = np.array(oldhypo, dtype=np.float128)
	newhypo = 1.0/(1.0+np.exp(-oldhypo))	
	return newhypo.T #np.array(newhypo.T, dtype=np.float64)

# Calculate the Hypothesis (layer 3) using just layer 1.
def ForwardProp(WAll, a1):
	W1, W2, b1, b2 = unLinWAll(WAll)
	# Calculate a2 (g.m x 25)
	a2 = logHypo(W1, b1, a1)
	# Calculate and return the output from a2 and W2 (g.m x 64)
	a3 = softHypo(W2, b2, a2)
	return a2, a3, W1, W2, b1, b2


# Calculate the regularized Cost J(theta)
def RegJCost(WAll, a1, ymat):
	# Forward Propagate
	a2, a3, W1, W2, b1, b2 = ForwardProp(WAll, a1)	# a2 (g.m x g.f2), a3 (g.m x g.f3)
	# Calculate J(W, b). ymat and a3 are the same shape: 15298 x 10
	J = (-1.0 / g.m)*np.sum( np.multiply(np.log(a3), ymat)  ) 
	J += g.lamb*0.5*( np.sum(W1**2)+np.sum(W2**2) )
	return J


# Calculate the gradient of cost function for all values of W1, W2, b1, and b2
def BackProp(WAll, a1, ymat):
	# To keep track of how many times this code is called
	g.step += 1
	if g.step % 50 == 0:
		print 'Global Step: ', g.step, 'with JCost: ',  RegJCost(WAll, a1, ymat)
	if g.step % 200 == 0:
		print 'Saving Global Step : ', g.step
		saveW(WAll)

	# Forward Propagate and extract the W matrices
	a2, a3, W1, W2, b1, b2 = ForwardProp(WAll, a1)	# a2 (g.m x g.f2), a3 (g.m x g.f3)

	# Now, to get backprop to work, I had to remake the theta matrices we had previously.
	# Sandwich b2 onto W2 and b1 onto W1
	theta2 = np.hstack((b2, W2))
	theta1 = np.hstack((b1, W1))
	# Attach a column of 1's onto a2 and a3
	a2ones = np.hstack(( np.ones((a2.shape[0],1)), a2 ))
	a1ones = np.hstack(( np.ones((a1.shape[0],1)), a1 ))
	# Calculate delta 3 and delta 2
	delta3 = ymat - a3		# 10 x 4
	gPrime = np.multiply(a2, 1 - a2)			# g.n x 37
	delta2 = np.multiply(np.matmul(delta3, W2), gPrime)		# g.n x 37

#	# Calculate the capital Deltas
#	Delta2 = np.zeros((g.f3, g.f2+1))			# 4 x 37
#	Delta1 = np.zeros((g.f2, g.f1+1))			# 36 x 3601

	Delta2 = np.matmul(delta3.T, a2ones)
	Delta1 = np.matmul(delta2.T, a1ones)
	# Calculate the derivatives
	D2 = -Delta2 / (g.m+0.0) + g.lamb * theta2	# 10 x 26
	D1 = -Delta1 / (g.m+0.0) + g.lamb * theta1	# 36 x 3601
	
	# Split it inearize it and send it
	return Lin4(D1[:, 1:], D2[:, 1:], D1[:, 0], D2[:, 0])


## I am trying to keep the features with their 3 x 3 matrix here, 
## so the first image's first 9 values (a1[0,:8]) is the linearized 3x3 matrix for the first feature
#def dat2D(cpdat):
#	a1 = np.zeros((cpdat.shape[1], 400*3*3))
#	for i in range(cpdat.shape[1]):
#		# Create a temporary matrix holding linearized data for each image and feature
#		temp = np.zeros((cpdat.shape[0], 9))
#		for j in range(cpdat.shape[0]):
#			temp[j,:] = np.ravel(cpdat[j,i,:,:])
#		a1[i] = np.ravel(temp)
#	return a1


# Generate the y-matrix. This is called only once, so I use loops
def GenYMat(yvals):
	yvals = np.ravel(yvals)
	yArr = np.zeros((len(yvals), g.f3))
	for i in range(len(yvals)):
		for j in range(g.f3):
			if yvals[i] == j:
				yArr[i][j] = 1
	return yArr



#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE



# To see how long the code runs for, we start a timestamp
totStart = time.time()



# DATA PROCESSING
# Get the convolved and pooled images. 
print "Grabbing the convolved and pooled data..."
# These are stored in 40 files, so I stitch them together
print 'Grabbing file: 1'
cpdat = np.genfromtxt(datStr, dtype=float)
#cpdat = np.genfromtxt(datStr+'1.out', dtype=float)
#for i in range(g.numfiles-1):
#	print 'Grabbing file:', i+2
#	cpdat = np.concatenate(( cpdat, np.genfromtxt(datStr+str(i+2)+'.out', dtype=float) ))

cpdat = cpdat.reshape((100, 60000, g.pooldim, g.pooldim))[:,:g.m,:,:]
# We need this in 2D form. g.100m x 3600  (3 x 3 x 400 = 3600)
a1 = np.swapaxes(cpdat, 0, 1)
a1 = a1.reshape(g.m, g.f1)
print "Got the data"
print ' '

# Get the y labels. Labeled in set [0, 9]
x, y = dataPrepdigit.GenTrain()
# Fix the label range to [0,3]
y = np.ravel(y[:g.m])
# Generate the y matrix. # 15298 x 10
ymat = GenYMat(y)



# INITIALIZING W MATRICES
# Prepare the W matrices and b vectors and linearize them
W1 = randMat(g.f2, g.f1)
W2 = randMat(g.f3, g.f2)
b1 = randMat(g.f2, 1)
b2 = randMat(g.f3, 1)
WAll = Lin4(W1, W2, b1, b2) # 1D vector, probably length 3289



# CALCULATING IDEAL W MATRICES
# Check the cost of the initial W matrices
print 'Initial W JCost: ', RegJCost(WAll, a1, ymat) #,BackProp(WAll, a1, ymat)

# Check the gradient. Go up and uncomment the import check_grad to use. ~6.38411247537693e-05 for 100 for randomized Ws and bs w/ g.f2=20
#print 'Grad Check:', check_grad(RegJCost, BackProp, WAll, a1, ymat)

res = minimize(fun=RegJCost, x0= WAll, method='L-BFGS-B', tol=10**g.tolexp, jac=BackProp, args=(a1, ymat) ) # options = {'disp':True}
bestWAll = res.x

print 'Final W JCost', RegJCost(bestWAll, a1, ymat)
saveW(bestWAll)



# Stop the timestamp and print out the total time
totend = time.time()
print ' '
print'cnn.py took ', totend - totStart, 'seconds to run'


