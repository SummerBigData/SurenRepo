# Written by: 	Suren Gourapura
# Written on: 	May 25, 2018
# Purpose: 	To solve exercise 4 on Multi-class Classification and Neural Networks in Coursera
# Goal:		Use backpropagation to find the best theta values

# Import the modules
import numpy as np
from math import exp, log
from scipy.optimize import minimize
import scipy.io
import time
import struct as st
import gzip
#import matplotlib.pyplot as plt
from scipy.optimize import check_grad



#----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE



gStep = 0

# These are the global constants used in the code
def g(char):
	if char == 'n':		# number of data points (number of 'number' pictures)
		return 30000 	# CHANGE THIS TO ADJUST TRAINING SET SIZE (up to 60,000)
	if char == 'f1':	# number of features (pixels)
		return 784
	if char == 'f2':	# number of features (hidden layer)
		return 36
	if char == 'lamb':	# the 'overfitting knob'
		return 0.1
	if char == 'eps':	# used for generating random theta matrices
		return 0.12
	if char == 'saveThetas':
		np.savetxt('thetaArrs/theta300MNIST-3L0.1.out', thetaAll, delimiter=',')

# Read the MNIST dataset
def read_idx(filename, n=None):
	with gzip.open(filename) as f:
		zero, dtype, dims = st.unpack('>HBB', f.read(4))
		shape = tuple(st.unpack('>I', f.read(4))[0] for d in range(dims))
		arr = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
		if not n is None:
			arr = arr[:n]
		return arr


# Calculate the Hypothesis (for layer l to l+1)
def hypothesis(thetaArr, xArr):
	oldhypo = np.matmul(thetaArr, np.transpose(xArr) )
	oldhypo = np.array(oldhypo, dtype=np.float128)
	newhypo = 1.0/(1.0+np.exp(-oldhypo))	
	return np.array(newhypo, dtype=np.float64)


# Calculate the Sigmoid's gradient UNUSED IN CODE
def sigmoidGrad(z):
	gz = 1/(1 + np.exp(-z))
	return gz*(1 - gz)


# Generate random theta matrices with a range [-eps, eps]
def randTheta(x, y):
	theta = np.random.rand(x,y) 	# Makes a (x) x (y) random matrix of [0,1]
	return theta*2*g('eps') - g('eps')	# Make it range [-eps, eps]


# Calculate the Hypothesis (layer 3) using just layer 1. xArr.shape -> 5000 x 401
def FullHypo(theta1, theta2, xArr):
	# Calculate a2 using theta1 and the x data (xArr == a1)
	a2 = hypothesis(theta1, xArr)
	# Add a row of 1's on top of the a1 matrix
	a2Arr = np.vstack(( np.asarray([1 for i in range(g('n'))]) , a2))
	# Calculate and return the output from a2 and theta2. This is a3 (a3.shape -> 10 x 5000)
	return hypothesis(theta2, a2Arr.T)


# Calculate the regularized Cost J(theta) using ALL thetas. yArr.shape = 5000 x 10
# Note: both thetas are fed in as one vector, need to be seperated and reshaped 
def RegJCost(thetaAll, xArr, yArr):
	#start = time.time()
	# Seperate and reshape the Theta values
	theta1, theta2 = UnLin(thetaAll, g('f2'), g('f1')+1, 10, g('f2')+1)

	hypo = FullHypo(theta1, theta2, xArr)
	J = (1.0/g('n'))*np.sum(   -1*np.multiply(yArr, np.log(hypo.T)) 
		- np.multiply(1-yArr, np.log(1 - hypo.T))      )
	
	J = J + (0.5 * g('lamb')/g('n'))*(   np.sum(theta1**2) - np.sum(column(theta1, 0)**2) + 
		np.sum(theta2**2) - np.sum(column(theta2, 0)**2) )

	global gStep
	if gStep % 200 == 0:
		print 'Saving Global Step : ', gStep
		g('saveThetas')

	#end = time.time()
	#print('RegJCost', end - start)
	return J


# Create the y matrix that is 5000 x 10, with a 1 in the index(number) and 0 elsewhere
# Also fixes 10 -> 0. Since this function is rarely called, I use loops
def GenYMat(yvals):
	yvals = np.ravel(yvals)
	yArr = np.zeros((g('n'), 10))
	for i in range(len(yvals)):
		for j in range(10):
			if yvals[i] == j or (yvals[i] == 10 and j == 0):
				yArr[i][j] = 1
	return yArr


def BackProp(thetaAll, xArr, yArr):
	# To keep track of how many times this code is called
	global gStep
	gStep += 1
	if gStep % 50 == 0:
		print 'Global Step: ', gStep

	#print 'BackProp iter. Theta[0] = ', thetaAll[0]
	#start = time.time()

	# Seperate and reshape the Theta values
	theta1, theta2 = UnLin(thetaAll, g('f2'), g('f1')+1, 10, g('f2')+1)

	# Forward propagating to get a2, a3
	a2 = hypothesis(theta1, xArr)					# 25 x 5000
	a2 = np.vstack(( np.asarray([1 for i in range(g('n'))]) , a2))	# Reshape to 26 x 5000 (Add bias row)
	a3 = FullHypo(theta1, theta2, xArr)				# 10 x 5000
	
	# Creating (Capital) Delta matrices
	Delta2 = np.zeros((10, g('f2')+1))			# 10 x 26
	Delta1 = np.zeros((g('f2'), g('f1')+1))			# 25 x 401
	
	# Calculate (Lowercase) deltas for each element in the dataset and add it's contributions to the Deltas
	for t in range(g('n')):
		delta3 = column(a3, t) - yArr[t]	# 10 vec
		gPrime = np.multiply(column(a2, t), 1 - column(a2, t))		# 26 vec
		delta2 = np.multiply(np.matmul(theta2.T, delta3), gPrime)	# 26 vec

		Delta2 = Delta2 + np.outer(column(a2, t), delta3).T
		# Note: we use the bias row to calculate Delta2 but not Delta1, so it is removed below
		Delta1 = Delta1 + np.delete( np.outer(column(xArr.T, t), delta2).T , 0, 0)	

	# Since the calculation calls for [if j=0, D = Delta/m], we make Theta matrices so [Theta(j=0)=1 for all i].
	Theta2 = np.delete(theta2, 0, 1)	# Remove the bias layers
	Theta1 = np.delete(theta1, 0, 1)	# Now, these are 10 x 25, 25 x 400 respectively

	Theta2 = np.hstack(( np.asarray([[0] for i in range(10)]) , Theta2))	# Add the bias layer as 0's
	Theta1 = np.hstack(( np.asarray([[0] for i in range(g('f2'))]) , Theta1))	# Now, these are 10 x 26, 25 x 401 respectively
	# Now we calculate D normally and the j = 0 case is taken care of
	D2 = (Delta2 + g('lamb') * Theta2) / (g('n')+0.0) 		# 10 x 26
	D1 = (Delta1 + g('lamb') * Theta1) / (g('n')+0.0)		# 25 x 401

#	D2 = (Delta2 ) / (n+0.0) 		10 x 26
#	D1 = (Delta1 ) / (n+0.0)		25 x 401


	DAll = Lin(D1, D2)
	#print DAll[0]
	#end = time.time()
	#print('BackProp', end - start)

	return DAll


# Take out one column (column i) from a matrix
def column(matrix, i):
    return np.asarray([row[i] for row in matrix])


# Linearize: Take 2 matrices, unroll them, and stitch them together into a vector
def Lin(a, b):
	return np.concatenate((np.ravel(a), np.ravel(b)))


# Unlinearize: Take a vector, break it into two vectors, and roll it back up
def UnLin(vec, a1, a2, b1, b2):
	if a1*a2+b1*b2 != len(vec):
		return 'Incorrect Dimensions! ', a1*a2+b1*b2, ' does not equal ', len(vec)

	else:		
		a = vec[0:a1*a2]
		b = vec[a1*a2:len(vec)]
		return np.reshape(a, (a1, a2)) , np.reshape(b, (b1, b2))


## Take the xvals matrix and extract the first or last instances of each number so that the total is now g('n') training samples
#def trunc(xvals, yvals, pos):	
#	f = g('n') / 10					# Pick out n/10 instances for each number
#	if pos == 'first':				# If the user wants the first 'f' values
#		xVals = xvals[0:f, 0:g('f1')]		# Put the zeros in
#		yVals = yvals[0:f]
#		for i in range(9):			# Put in the rest
#			xVals = np.append(xVals, xvals[500*(i+1) : f+500*(i+1), 0:g('f1')], axis=0)
#			yVals = np.append(yVals, yvals[500*(i+1) : f+500*(i+1)])
#	if pos == 'last':				# If the user wants the last 'f' values
#		xVals = xvals[500-f:500, 0:g('f1')]	# Put the zeros in
#		yVals = yvals[500-f:500]
#		for i in range(9):			# Put in the rest
#			xVals = np.append(xVals, xvals[500*(i+2)-f : 500*(i+2), 0:g('f1')], axis=0)
#			yVals = np.append(yVals, yvals[500*(i+2)-f : 500*(i+2)])
#	return xVals, yVals


# Reorder the data randomly
def randData(xvals, yvals):
	# Reshape the y values into a column, so hstack can work
	yvals = np.column_stack(yvals).T
	# Combine the x and y values so that the shuffling doesn't seperate them
	XandY = np.hstack((xvals, yvals))
	# Shuffle the matrix. We are shuffling the rows here
	np.random.shuffle(XandY)
	# Reseperate the matrices
	xVals = XandY[0:g('n'),0:g('f1')]
	yVals = XandY[0:g('n'),g('f1'):g('f1')+1]
	# Convert the y values back as integers and as vectors, not matrices
	yVals = np.ravel(yVals.astype(int))
	return xVals, yVals



#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE



# To see how long the code runs for, we start a timestamp
totStart = time.time()


# PREPARING DATA
# Obtain the data values and convert them from arrays to lists
datx = read_idx('data/train-images-idx3-ubyte.gz', g('n'))
daty = read_idx('data/train-labels-idx1-ubyte.gz', g('n'))

print datx.shape, daty.shape

datx = np.ravel(datx).reshape((g('n'), g('f1')))
print datx.shape

# Reorder the data randomly
#datx, daty = randData(datx, daty)

# Form the correct x and y arrays, with a column of 1's in the xArr
xArr = np.hstack(( np.asarray([[1] for i in range(g('n'))]) , datx))	# g('n') x g('f1')
yArr = GenYMat(daty)							# g('n') x 10
				
# Randomize theta1 and theta2. Comment this out to use their theta values (weights)
theta1 = randTheta(g('f2'), g('f1')+1)
theta2 = randTheta(10, g('f2')+1)

# Reshape and splice theta1, theta2
thetaAll = Lin(theta1, theta2)



#Show a random set of numbers in the dataset. Go up and include plt
#pic0 = np.reshape(datx[0], (28,28))
#pic1 = np.reshape(datx[1], (28,28))
#pic2 = np.reshape(datx[2], (28,28))
#pic3 = np.reshape(datx[3], (28,28))
#pic4 = np.reshape(datx[4], (28,28))
#pic5 = np.reshape(datx[5], (28,28))
#pic6 = np.reshape(datx[6], (28,28))
#pic7 = np.reshape(datx[7], (28,28))
#pic8 = np.reshape(datx[8], (28,28))
#pic9 = np.reshape(datx[9], (28,28))
## Stitch these all together into one picture
#picAll = np.concatenate((pic0, pic1, pic2, pic3, pic4, pic5, pic6, pic7, pic8, pic9), axis = 1)
#imgplot = plt.imshow(picAll, cmap="binary", interpolation='none') 
#plt.show()



# MINIMIZING THETAS
# Check the cost of the initial theta matrices
print 'Initial Theta JCost: ', RegJCost(thetaAll, xArr, yArr)  # Outputting 10.537, not 0.38377 for their theta matrices

# Check the gradient function. ~1.0405573537e-05 for randomized thetas
# print check_grad(RegJCost, BackProp, thetaAll, xArr, yArr)

# Calculate the best theta values for a given j and store them. Usually tol=10e-4
res = minimize(fun=RegJCost, x0= thetaAll, method='CG', tol=10e-3, jac=BackProp, args=(xArr, yArr))
bestThetas = res.x

print 'Final Theta JCost', RegJCost(bestThetas, xArr, yArr)

g('saveThetas')

# Stop the timestamp and print out the total time
totend = time.time()
print'neuTrainerMNIST.py took ', totend - totStart, 'seconds to run'





