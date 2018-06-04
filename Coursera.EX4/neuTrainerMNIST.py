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
# For imgNorm
from numpy.polynomial import polynomial as P
from scipy.ndimage import rotate
from math import pi, atan
# This will be fun
import argparse



#---------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES



parser = argparse.ArgumentParser()
parser.add_argument("n", help="Number of Datapoints", type=int)
parser.add_argument("f1", help="Number of Features (pixels) in images", type=int)
parser.add_argument("f2", help="Number of Features in hidden layer", type=int)
parser.add_argument("lamb", help="Lambda, the overfitting knob", type=float)
parser.add_argument("eps", help="Bounds for theta matrix randomization, [-eps, eps]", type=float)
parser.add_argument("tolexp", help="Exponent of tolerance of minimize function, good value 10e-4, so -4", type=int)
parser.add_argument("normImg", help="Choose whether or not to straighten the images", type=bool)
g = parser.parse_args()
saveStr = 'thetaArrs/theta' + str(g.n)+ 'MNIST'+str(g.tolexp)+'Lamb'+str(g.lamb)+'.out'
gStep = 0

print 'You have chosen:', g
print 'Will be saved in: ', saveStr

#----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE



# Save an instance of theta values
def saveTheta(theta):
	np.savetxt(saveStr, theta, delimiter=',')


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


# Generate random theta matrices with a range [-eps, eps]
def randTheta(x, y):
	theta = np.random.rand(x,y) 	# Makes a (x) x (y) random matrix of [0,1]
	return theta*2*g.eps - g.eps	# Make it range [-eps, eps]


# Calculate the Hypothesis (layer 3) using just layer 1. xArr.shape -> 5000 x 401
def FullHypo(theta1, theta2, xArr):
	# Calculate a2 using theta1 and the x data (xArr == a1)
	a2 = hypothesis(theta1, xArr)
	# Add a row of 1's on top of the a1 matrix
	a2Arr = np.vstack(( np.asarray([1 for i in range(g.n)]) , a2))
	# Calculate and return the output from a2 and theta2. This is a3 (a3.shape -> 10 x 5000)
	return hypothesis(theta2, a2Arr.T)


# Calculate the regularized Cost J(theta) using ALL thetas. yArr.shape = 5000 x 10
# Note: both thetas are fed in as one vector, need to be seperated and reshaped 
def RegJCost(thetaAll, xArr, yArr):
	#start = time.time()
	# Seperate and reshape the Theta values
	theta1, theta2 = UnLin(thetaAll, g.f2, g.f1+1, 10, g.f2+1)

	hypo = FullHypo(theta1, theta2, xArr)
	J = (1.0/g.n)*np.sum(   -1*np.multiply(yArr, np.log(hypo.T)) 
		- np.multiply(1-yArr, np.log(1 - hypo.T))      )
	
	J = J + (0.5 * g.lamb/g.n)*(   np.sum(theta1**2) - np.sum(column(theta1, 0)**2) + 
		np.sum(theta2**2) - np.sum(column(theta2, 0)**2) )

	#end = time.time()
	#print('RegJCost', end - start)
	return J


# Create the y matrix that is 5000 x 10, with a 1 in the index(number) and 0 elsewhere
# Also fixes 10 -> 0. Since this function is rarely called, I use loops
def GenYMat(yvals):
	yvals = np.ravel(yvals)
	yArr = np.zeros((g.n, 10))
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
	if gStep % 200 == 0:
		print 'Saving Global Step : ', gStep
		saveTheta(thetaAll)

	#print 'BackProp iter. Theta[0] = ', thetaAll[0]
	#start = time.time()

	# Seperate and reshape the Theta values
	theta1, theta2 = UnLin(thetaAll, g.f2, g.f1+1, 10, g.f2+1)

	# Forward propagating to get a2, a3
	a2 = hypothesis(theta1, xArr)					# 25 x 5000
	a2 = np.vstack(( np.asarray([1 for i in range(g.n)]) , a2))	# Reshape to 26 x 5000 (Add bias row)
	a3 = FullHypo(theta1, theta2, xArr)				# 10 x 5000
	
	# Creating (Capital) Delta matrices
	Delta2 = np.zeros((10, g.f2+1))			# 10 x 26
	Delta1 = np.zeros((g.f2, g.f1+1))			# 25 x 401
	
	# Calculate (Lowercase) deltas for each element in the dataset and add it's contributions to the Deltas
	for t in range(g.n):
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
	Theta1 = np.hstack(( np.asarray([[0] for i in range(g.f2)]) , Theta1))	# Now, these are 10 x 26, 25 x 401 respectively
	# Now we calculate D normally and the j = 0 case is taken care of
	D2 = (Delta2 + g.lamb * Theta2) / (g.n+0.0) 		# 10 x 26
	D1 = (Delta1 + g.lamb * Theta1) / (g.n+0.0)		# 25 x 401

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
	xVals = XandY[0:g.n,0:g.f1]
	yVals = XandY[0:g.n,g.f1:g.f1+1]
	# Convert the y values back as integers and as vectors, not matrices
	yVals = np.ravel(yVals.astype(int))
	return xVals, yVals


def normImg(datx):
	print "Normalizing data"
	# Get the size of the picture matrices
	s = int(np.sqrt(g.f1))
	# Create an array of numbers [1,2,3,4,...] to use for average position computation
	index = np.zeros((s))
	for i in range(s):
		index[i] = i + 1
	# Calculate the rotated matrix for all data points. First, initialize it
	rotmat = np.zeros((g.n, g.f1))

	for i in range(g.n):
		# Convert it back to a matrix
		mat = np.reshape(datx[i], (s,s))
		hcenter = np.zeros((s))
		# We need the horizontal centers for each row
		for j in range(s):
			# Handle the zero case seperately, due to divide by zero. The value here doesn't matter, since the weight will kill it
			if sum(mat[j]) == 0:
				hcenter[j] = -1
			# Calculate and store the center of each column
			else:
				hcenter[j] = sum(mat[j]*index)/ (sum(mat[j])+0.0)
		# We don't want to include the zero cases, so form a weights matrix to record them
		weights = np.zeros((s))
		for j in range(s):
			if hcenter[j] < 0:
				weights[j] = 0
			else:
				weights[j] = 1
#		print hcenter
		# Calculate the line of best fit for all of the horizontal centers
		c = P.polyfit(index,hcenter,1,full=False, w=weights)
		# Here's some tools to visualize the process
#		print c[0], c[1], atan(c[1])*180.0/pi
#		bestfit = c[0] + c[1]*index
#		plt.plot(hcenter,'green', bestfit, 'red')
#		plt.show()
		# Rotate, unravel, and record the matrix
	
		rotmat[i] = np.ravel(rotate(mat, -1*atan(c[1])*180.0/pi, reshape=False)).reshape(1, g.f1)
	
	
	return rotmat




#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE



# To see how long the code runs for, we start a timestamp
totStart = time.time()


# PREPARING DATA
# Obtain the data values and convert them from arrays to lists
datx = read_idx('data/train-images-idx3-ubyte.gz', g.n)
daty = read_idx('data/train-labels-idx1-ubyte.gz', g.n)
	
datx = np.ravel(datx).reshape((g.n, g.f1))

# Straighten the images if desired:
if normImg == True:
	datx = normImg(datx)

##Show a random set of numbers in the dataset. Go up and include plt
#pic0 = np.reshape(datx[10], (28,28))
#pic1 = np.reshape(datx[11], (28,28))
#pic2 = np.reshape(datx[12], (28,28))
#pic3 = np.reshape(datx[13], (28,28))
#pic4 = np.reshape(datx[14], (28,28))
#pic5 = np.reshape(datx[15], (28,28))
#pic6 = np.reshape(datx[16], (28,28))
#pic7 = np.reshape(datx[17], (28,28))
#pic8 = np.reshape(datx[18], (28,28))
#pic9 = np.reshape(datx[19], (28,28))

#pic0f = np.reshape(datX[10], (28,28))
#pic1f = np.reshape(datX[11], (28,28))
#pic2f = np.reshape(datX[12], (28,28))
#pic3f = np.reshape(datX[13], (28,28))
#pic4f = np.reshape(datX[14], (28,28))
#pic5f = np.reshape(datX[15], (28,28))
#pic6f = np.reshape(datX[16], (28,28))
#pic7f = np.reshape(datX[17], (28,28))
#pic8f = np.reshape(datX[18], (28,28))
#pic9f = np.reshape(datX[19], (28,28))
## Stitch these all together into one picture
#picAll1 = np.concatenate((pic0, pic1, pic2, pic3, pic4, pic5, pic6, pic7, pic8, pic9), axis = 1)
#picAll2 = np.concatenate((pic0f, pic1f, pic2f, pic3f, pic4f, pic5f, pic6f, pic7f, pic8f, pic9f), axis = 1)
#picAll = np.vstack((picAll1, picAll2))
#imgplot = plt.imshow(picAll, cmap="binary", interpolation='none') 
#plt.show()


# Reorder the data randomly
#datx, daty = randData(datx, daty)

# Form the correct x and y arrays, with a column of 1's in the xArr
xArr = np.hstack(( np.asarray([[1] for i in range(g.n)]) , datx))	# g('n') x g('f1')+1
yArr = GenYMat(daty)							# g('n') x 10
				
# Randomize theta1 and theta2. Comment this out to use their theta values (weights)
theta1 = randTheta(g.f2, g.f1+1)
theta2 = randTheta(10, g.f2+1)

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
res = minimize(fun=RegJCost, x0= thetaAll, method='CG', tol=10**g.tolexp, jac=BackProp, args=(xArr, yArr))
bestThetas = res.x

print 'Final Theta JCost', RegJCost(bestThetas, xArr, yArr)

saveTheta(bestThetas)

# Stop the timestamp and print out the total time
totend = time.time()
print'neuTrainerMNIST.py took ', totend - totStart, 'seconds to run'





