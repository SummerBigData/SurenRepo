# Written by: 	Suren Gourapura
# Written on: 	May 22, 2018
# Purpose: 	To solve exercise 3 on Multi-class Classification and Neural Networks in Coursera
# Goal:		?

# Import the modules
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import exp, log
from scipy.optimize import minimize
import scipy.io


#----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE


# Calculate the Hypothesis
def hypothesis(thetaArr, xArr):
	oldhypo = np.matmul(thetaArr, np.transpose(xArr) )
	newhypo = 1/(1+np.exp(-oldhypo))
	return newhypo

# Calculate the Cost J(theta)
def JCost(thetaArr, xArr, yArr):
	hypo = hypothesis(thetaArr, xArr)
	J = sum(  (1.0/len(yArr))*(  -1*yArr*np.log(hypo) - (1-yArr)*np.log(1 - hypo)  )  )
	return J

# Calculate the gradJ vector
def gradJ(thetaArr, xArr, yArr):
	hypo = hypothesis(thetaArr, xArr)
	gradj = (1.0/len(yArr))*np.matmul( np.transpose(xArr), hypo - yArr)
	return gradj	

# Calculate the regularized Cost J(theta)
def RegJCost(thetaArr, xArr, yArr, lamb):
	hypo = hypothesis(thetaArr, xArr)
	J = sum(  (1.0/len(yArr))*(  -1*yArr*np.log(hypo) - (1-yArr)*np.log(1 - hypo)  )  )
	J = J + (0.5 * lamb/len(yArr))*(  sum(thetaArr**2) - thetaArr[0]**2 )
	return J

# Calculate the regularized gradJ vector
def RegGradJ(thetaArr, xArr, yArr, lamb):
	hypo = hypothesis(thetaArr, xArr)
	gradj = (1.0/len(yArr))*np.matmul( np.transpose(xArr), hypo - yArr)
	gradj = gradj + (lamb/len(yArr))*thetaArr
	gradj[0] = gradj[0] - (lamb/len(yArr))*thetaArr[0]
	return gradj

# Create an array that stores 1 if the wanted number is at that index, and 0 otherwise
def MakeY(yvals, number):
	yArr = np.asarray([0 for i in range(len(yvals))])
	# Since the data has the 0 digits labeled as 10:
	if number == 0:
		number = 10
	# Go through the y-values and store yArr[i]=1 iff y = number, 0 otherwise
	for i in range(n):
		if yvals[i] == number:
			yArr[i] = 1
		else:
			yArr[i] = 0
	return yArr

# Take out one column (column i) from a matrix
def column(matrix, i):
    return [row[i] for row in matrix]


#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE
	

# Obtain the data values and convert them from arrays to lists
data = scipy.io.loadmat('ex3data1.mat')
xvals = data['X']
yvals = data['y']
n = 5000	# number of data points (number of 'number' pictures)
f = 400		# number of features (pixels)
lamb = 10	# regularization variable: high = underfit, low = overfit

# Form the correct x array, with xArr[0:5000, 0:1] being a column of 1's
xArr = np.hstack(( np.asarray([[1] for i in range(n)]) , xvals))

# We need to store the ideal theta values for each case of one-vs-all, so thetaAll is a 10 x 401 matrix
thetaAll = np.asarray( [[0.0 for i in range(f+1)] for j in range(10)] )

#print RegJCost(thetaAll[0], xArr, MakeY(yvals, 0), lamb)

# We need to also store the best guess calculated by each theta value, so we use a 10 x 5000 matrix
guessAll = np.array( [[0.0 for i in range(n)] for j in range(10)] )

for j in range(10):

	# Form a solution array, with y=1 for j values and y=0 for the rest
	yArr = MakeY(yvals, j)

	# Calculate the best theta values for a given j and store them. BFGS
	res = minimize(fun=RegJCost, x0= thetaAll[j], method='CG', jac=RegGradJ, args=(xArr,yArr,lamb) )
	thetaAll[j] = res.x

	# Calculate the best guess given the best theta values
	guessAll[j] = hypothesis(thetaAll[j], xArr)
	
	print 'Calculating best theta, loop: ', j	

# Now we need to look through the hypothesis matrix and find the highest value in each column.
# This collapses the guessAll 10 x 5000 matrix into a guessBest 5000 1D array
guessBest = np.asarray([0 for i in range(n)])

for j in range(n):
	tempGuessAll = np.ravel(column(guessAll,j))
	guessBest[j] = tempGuessAll.argmax()

# Now we want to see what percent of each number the code got right
numPercent = np.array([0.0 for i in range(10)])

for i in range(n):
	if guessBest[i] == yvals[i]:
		numPercent[ yvals[i] ] = numPercent[ yvals[i] ] + 1.0

	if guessBest[i] == 0 and yvals[i]==10:
		numPercent[0] = numPercent[0] + 1.0

numPercent = numPercent * (1/ 500.0)

print numPercent


