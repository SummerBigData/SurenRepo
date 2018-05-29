# Written by: 	Suren Gourapura
# Written on: 	May 25, 2018
# Purpose: 	To solve exercise 4 on Multi-class Classification and Neural Networks in Coursera
# Goal:		Use backpropagation to find the best theta values (a complete Neural Network Code)

# Import the modules
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import exp, log
from scipy.optimize import minimize
import scipy.io


#----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE


# Calculate the Hypothesis (for layer l to l+1)
def hypothesis(thetaArr, xArr):
	oldhypo = np.matmul(thetaArr, np.transpose(xArr) )
	newhypo = 1/(1+np.exp(-oldhypo))
	return newhypo


# Calculate the Sigmoid's gradient UNUSED IN CODE
def sigmoidGrad(z):
	gz = 1/(1 + np.exp(-z))
	return gz*(1 - gz)



# Generate random theta matrices with a range [-eps, eps]
def randTheta(x, y, eps):
	theta = np.random.rand(x,y) 	# Makes a (x) x (y) random matrix of [0,1]
	return theta*2*eps - eps	# Make it range [-eps, eps]



# Calculate the Hypothesis (layer 3) using just layer 1. xArr.shape -> 5000 x 401
def FullHypo(theta1, theta2, xArr, n):
	# Calculate a2 using theta1 and the x data (xArr == a1)
	a2 = hypothesis(theta1, xArr)
	# Add a row of 1's on top of the a1 matrix
	a2Arr = np.vstack(( np.asarray([1 for i in range(n)]) , a2))
	# Calculate and return the output from a2 and theta2. This is a3 (a3.shape -> 10 x 5000)
	return hypothesis(theta2, a2Arr.T)



# Calculate the regularized Cost J(theta) using ALL thetas. yArr.shape = 5000 x 10
# Note: both thetas are fed in as one vector, need to be seperated and reshaped 
def RegJCost(thetaAll, xArr, yArr, lamb, n):
	# Seperate and reshape the Theta values
	theta1, theta2 = UnLin(thetaAll, 25, 401, 10, 26)

	hypo = FullHypo(theta1, theta2, xArr, n)
	J = (1.0/n)*np.sum(   -1*np.multiply(yArr, np.log(hypo.T)) 
		- np.multiply(1-yArr, np.log(1 - hypo.T))      )
	J = J + (0.5 * lamb/n)*(   np.sum(theta1**2) - np.sum(column(theta1, 0)**2) + 
		np.sum(theta2**2) - np.sum(column(theta2, 0)**2) )
	return J



# Create the y matrix that is 5000 x 10, with a 1 in the index(number) and 0 elsewhere
# Also fixes 10 -> 0. Since this function is rarely called, I use loops
def GenYMat(yvals, n):
	success = 0
	yvals = np.ravel(yvals)
	yArr = np.zeros((n, 10))
	for i in range(len(yvals)):
		for j in range(10):
			if yvals[i] == j or (yvals[i] == 10 and j == 0):
				yArr[i][j] = 1
	return yArr



def BackProp(thetaAll, xArr, yArr, lamb, n):

	print 'BackProp iter. Theta[0] = ', thetaAll[0]

	# Seperate and reshape the Theta values
	theta1, theta2 = UnLin(thetaAll, 25, 401, 10, 26)

	# Forward propagating to get a2, a3
	a2 = hypothesis(theta1, xArr)					# 25 x 5000
	a2 = np.vstack(( np.asarray([1 for i in range(n)]) , a2))	# Reshape to 26 x 5000 (Add bias row)
	a3 = FullHypo(theta1, theta2, xArr, n)				# 10 x 5000
	
	# Creating (Capital) Delta matrices
	Delta2 = np.zeros((10, 26))			# 10 x 26
	Delta1 = np.zeros((25, 401))			# 25 x 401
	
	# Calculate (Lowercase) deltas for each element in the dataset and add it's contributions to the Deltas
	for t in range(n):
		delta3 = column(a3, t) - yArr[t]	# 10 vec
		gPrime = np.multiply(column(a2, t), 1 - column(a2, t))		# 26 vec
		delta2 = np.multiply(np.matmul(theta2.T, delta3), gPrime)	# 26 vec

		Delta2 = Delta2 + np.outer(column(a2, t), delta3).T
		# Note: we use the bias row to calculate Delta2 but not Delta1, so it is removed below
		Delta1 = Delta1 + np.delete( np.outer(column(xArr.T, t), delta2).T , 0, 0)	

#	# Since the calculation calls for [if j=0, D = Delta/m], we make Theta matrices so [Theta(j=0)=1 for all i].
#	Theta2 = np.delete(theta2, 0, 1)	# Remove the bias layers
#	Theta1 = np.delete(theta1, 0, 1)	# Now, these are 10 x 25, 25 x 400 respectively

#	Theta2 = np.hstack(( np.asarray([[0] for i in range(10)]) , Theta2))	# Add the bias layer as 1's
#	Theta1 = np.hstack(( np.asarray([[0] for i in range(25)]) , Theta1))	# Now, these are 10 x 26, 25 x 401 respectively
#	# Now we calculate D normally and the j = 0 case is taken care of
#	D2 = (Delta2 + lamb * Theta2) / (n+0.0) 	# 10 x 26
#	D1 = (Delta1 + lamb * Theta1) / (n+0.0)		# 25 x 401

	D2 = (Delta2 ) / (n+0.0) 	# 10 x 26
	D1 = (Delta1 ) / (n+0.0)		# 25 x 401


	DAll = Lin(D1, D2)
	print DAll[0]
	return DAll



def GradCheck(thetaAll, xArr, yArr, lamb, n):
	e = 10**(-4)
	Grad = np.zeros(len(thetaAll))

	for i in range(len(thetaAll)):
		if i%10 == 0:
			print 'GradCheck Iter. ', i
		eVec = np.zeros(len(thetaAll)-1)
		eVec = np.insert(eVec, i, e)
		tPlus = RegJCost(thetaAll + eVec, xArr, yArr, lamb, n)
		tMinus = RegJCost(thetaAll - eVec, xArr, yArr, lamb, n)
		Grad[i] = (tPlus - tMinus)/(2*e)
	return Grad





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




#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE
	

# Obtain the data values and convert them from arrays to lists
data = scipy.io.loadmat('ex4data1.mat')
weights = scipy.io.loadmat('ex4weights.mat')
xvals = data['X']		# 5000 x 400 
yvals = data['y']
theta1 = weights['Theta1']	# 25 x 401 matrix that takes x to a_1
theta2 = weights['Theta2']	# 10 x 26 matrix that takes a_1 to output (y)
n = 5000	# number of data points (number of 'number' pictures)
f = 400		# number of features (pixels)
lamb = 1	# the 'overfitting knob'
eps = 0.12	# used for generating random theta matrices

# Form the correct x and y arrays, with xArr[0:5000, 0:1] being a column of 1's
xArr = np.hstack(( np.asarray([[1] for i in range(n)]) , xvals))
yArr = GenYMat(yvals, n)

##print RegJCost(theta1, theta2, xArr, yArr, lamb, n) # Outputting 10.537, not 0.38377
#theta1 = randTheta(25, 401, eps)
#theta2 = randTheta(10, 26, eps)
# Reshape and splice theta1, theta2
thetaAll = Lin(theta1, theta2)


 
Cost = RegJCost(thetaAll, xArr, yArr, lamb, n)
print Cost

Grad = BackProp(thetaAll, xArr, yArr, lamb, n)
gradCheck = GradCheck(thetaAll, xArr, yArr, lamb, n)

print Grad[0:5]
print gradCheck[0:5]

## We need to also store the best guess calculated by each theta value, so we use a 10 x 5000 matrix
#guessAll = [0 for i in range(len(thetaAll))]

## Calculate the best theta values for a given j and store them. BFGS
#res = minimize(fun=RegJCost, x0= thetaAll, method='CG', jac=BackProp, args=(xArr, yArr, lamb, n))
#guessAll = res.x

#print guessAll.shape
#MinCost = RegJCost(guessAll, xArr, yArr, lamb, n)
#print MinCost
#guessTheta1, guessTheta2 = UnLin(guessAll, 25, 401, 10, 26)


## Calculate the best guess given the best theta values
#guessAll[j] = hypothesis(thetaAll[j], xArr)









## Calculate a1 using theta1 and the x data
#a1 = hypothesis(theta1, xArr)

## Add a row of 1's on top of the a1 matrix
#a1Arr = np.vstack(( np.asarray([1 for i in range(n)]) , a1))

## Calculate and store the output from a1 and theta2 (this contains the solutions)
#guessAll = hypothesis(theta2, a1Arr.T)

## We need to parse through this 10 x 5000, find the highest values, and record them in a 5000 1D array
#guessBest = np.asarray([0 for i in range(n)])
#for j in range(n):
#	tempGuessAll = np.ravel(column(guessAll,j))
#	guessBest[j] = tempGuessAll.argmax()	# Record the index of the highest value

## Print some sample sections where the guesses should transition, from 0-1, 1-2, 2-3, 8-9
#print guessBest[480:510], guessBest[980:1010], guessBest[1490:1510], guessBest[4490:4510]

## Now we want to see what percent of each number the code got right
#numPercent = np.array([0.0 for i in range(10)])

## The matrices report 0 as 9, 1 as 0, 2 as 1, 3 as 2, etc. To fix this, we hardcode a translation when calculating the percentages

#for i in range(n):
#	if yvals[i] == 10 and guessBest[i] == 9:
#		numPercent[0] = numPercent[0] + 1.0
#	elif guessBest[i]+1 == yvals[i]:
#		numPercent[ yvals[i] ] = numPercent[ yvals[i] ] + 1.0

#numPercent = numPercent * (1/ 500.0)
#print numPercent


# We also want to plot the a1 data as pictures, to see the hidden layer

# Generate an image. The image is inverted for some reason, so we transpose the matrix first
# We are plotting the first instance of each number in the data
# a1t = a1.T
#pic0 = np.transpose(np.reshape(a1t[1], (5, 5)))
#pic1 = np.transpose(np.reshape(a1t[500], (5, 5)))
#pic2 = np.transpose(np.reshape(a1t[1000], (5, 5)))
#pic3 = np.transpose(np.reshape(a1t[1500], (5, 5)))
#pic4 = np.transpose(np.reshape(a1t[2000], (5, 5)))
#pic5 = np.transpose(np.reshape(a1t[2500], (5, 5)))
#pic6 = np.transpose(np.reshape(a1t[3000], (5, 5)))
#pic7 = np.transpose(np.reshape(a1t[3500], (5, 5)))
#pic8 = np.transpose(np.reshape(a1t[4000], (5, 5)))
#pic9 = np.transpose(np.reshape(a1t[4500], (5, 5)))

#picAll = [ [0 for i in range(95)] for j in range(5)]

#for k in range(10):
#	pic0 = np.transpose(np.reshape(a1t[0+500*k], (5, 5)))
#	pic1 = np.transpose(np.reshape(a1t[1+500*k], (5, 5)))
#	pic2 = np.transpose(np.reshape(a1t[2+500*k], (5, 5)))
#	pic3 = np.transpose(np.reshape(a1t[3+500*k], (5, 5)))
#	pic4 = np.transpose(np.reshape(a1t[4+500*k], (5, 5)))
#	pic5 = np.transpose(np.reshape(a1t[5+500*k], (5, 5)))
#	pic6 = np.transpose(np.reshape(a1t[6+500*k], (5, 5)))
#	pic7 = np.transpose(np.reshape(a1t[7+500*k], (5, 5)))
#	pic8 = np.transpose(np.reshape(a1t[8+500*k], (5, 5)))
#	pic9 = np.transpose(np.reshape(a1t[9+500*k], (5, 5)))
#	space = np.asarray([ [0 for i in range(5)] for j in range(5)])

#	# Stitch these all together into one picture
#	picRow = np.concatenate((pic0, space, pic1, space, pic2, space, pic3, space, pic4, space, pic5, space, pic6, 			space, pic7, space, pic8, space, pic9), axis = 1)
#	
#	emptyRow = [[0 for i in range(95)] for j in range(5)]
#	
#	picAll = np.concatenate((picAll, picRow, emptyRow), axis = 0)

## 'binary' for black on white, 'gray' for white on black. 
## See https://matplotlib.org/examples/color/colormaps_reference.html for more color options

#imgplot = plt.imshow(picAll, cmap="binary") 
#plt.show()



