# Written by: 	Suren Gourapura
# Written on: 	May 29, 2018
# Purpose: 	To solve exercise 4 on Multi-class Classification and Neural Networks in Coursera
# Goal:		Use calculated theta values and calculate probabilities

# Import the modules
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import exp, log
import scipy.io


#----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE


# These are the global constants used in the code
def g(char):
	if char == 'n':		# number of data points (number of 'number' pictures)
		return 5000 #FIX
	if char == 'f1':	# number of features (pixels)
		return 400
	if char == 'f2':	# number of features (hidden layer)
		return 25
	if char == 'lamb':	# the 'overfitting knob'
		return 1
	if char == 'eps':	# used for generating random theta matrices
		return 0.12


# Calculate the Hypothesis (for layer l to l+1)
def hypothesis(thetaArr, xArr):
	oldhypo = np.matmul(thetaArr, np.transpose(xArr) )
	newhypo = 1/(1+np.exp(-oldhypo))
	return newhypo


# Calculate the Hypothesis (layer 3) using just layer 1. xArr.shape -> 5000 x 401
def FullHypo(theta1, theta2, xArr):
	# Calculate a2 using theta1 and the x data (xArr == a1)
	a2 = hypothesis(theta1, xArr)
	# Add a row of 1's on top of the a1 matrix
	a2Arr = np.vstack(( np.asarray([1 for i in range(g('n'))]) , a2))
	# Calculate and return the output from a2 and theta2. This is a3 (a3.shape -> 10 x 5000)
	return hypothesis(theta2, a2Arr.T)


# Unlinearize: Take a vector, break it into two vectors, and roll it back up
def UnLin(vec, a1, a2, b1, b2):
	if a1*a2+b1*b2 != len(vec):
		return 'Incorrect Dimensions! ', a1*a2+b1*b2, ' does not equal ', len(vec)

	else:		
		a = vec[0:a1*a2]
		b = vec[a1*a2:len(vec)]
		return np.reshape(a, (a1, a2)) , np.reshape(b, (b1, b2))


# Take out one column (column i) from a matrix
def column(matrix, i):
    return [row[i] for row in matrix]


# Take the xvals matrix and extract the first instances of each number so that the total is now g('n')
def trunc(xvals, yvals):	
	f = g('n') / 10			# Pick out n/10 instances for each number
	xVals = xvals[0:f, 0:400]	# Put the zeros in
	yVals = yvals[0:f]

	for i in range(9):
		xVals = np.append(xVals, xvals[500*(i+1) : f+500*(i+1), 0:400], axis=0)
		yVals = np.append(yVals, yvals[500*(i+1) : f+500*(i+1)])
	return xVals, yVals


# Create the y matrix that is 5000 x 10, with a 1 in the index(number) and 0 elsewhere
# Also fixes 10 -> 0. Since this function is rarely called, I use loops
def GenYMat(yvals):
	success = 0
	yvals = np.ravel(yvals)
	yArr = np.zeros((g('n'), 10))
	for i in range(len(yvals)):
		for j in range(10):
			if yvals[i] == j or (yvals[i] == 10 and j == 0):
				yArr[i][j] = 1
	return yArr



#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE



# PREPARING DATA
# Obtain the data values and convert them from arrays to lists
data = scipy.io.loadmat('ex4data1.mat')
xvals = data['X']		# 5000 x 400 
yvals = data['y']

bestThetas = np.genfromtxt('neuralThetas500.out', dtype=float)

# Truncate the data to a more manageable piece
#xvals, yvals = trunc(xvals, yvals)

# Form the correct x and y arrays, with xArr[0:5000, 0:1] being a column of 1's
xArr = np.hstack(( np.asarray([[1] for i in range(g('n'))]) , xvals))	# 5000 x 401
yArr = GenYMat(yvals)							# 5000 x 10
						






bestTheta1, bestTheta2 = UnLin(bestThetas, g('f2'), g('f1')+1, 10, g('f2')+1)


# Calculate the best guess given the best theta values
guesses = FullHypo(bestTheta1, bestTheta2, xArr)	#10 x 5000


# We need to parse through this 10 x 5000, find the highest values, and record them in a 5000 1D array
guessBest = np.asarray([0 for i in range(g('n'))])

for j in range(g('n')):
	tempGuessAll = np.ravel(column(guesses,j))
	guessBest[j] = tempGuessAll.argmax()	# Record the index of the highest value

# Print some sample sections where the guesses should transition, from 0-1, 1-2, 2-3, 8-9
print guessBest[480:510], guessBest[980:1010], guessBest[1490:1510], guessBest[4490:4510]

# Now we want to see what percent of each number the code got right
numPercent = np.array([0.0 for i in range(10)])

# The matrices report 0 as 9, 1 as 0, 2 as 1, 3 as 2, etc. To fix this, we hardcode a translation when calculating the percentages

for i in range(g('n')):
	if yvals[i] == 10 and guessBest[i] == 0:
		numPercent[0] = numPercent[0] + 1.0
	elif guessBest[i] == yvals[i]:
		numPercent[ yvals[i] ] = numPercent[ yvals[i] ] + 1.0

numPercent = numPercent * (10.0/ g('n'))
print numPercent





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

a1 = hypothesis(bestTheta1, xArr)
a1t = a1.T

picAll = [ [0 for i in range(95)] for j in range(5)]

for k in range(10):
	pic0 = np.transpose(np.reshape(a1t[0+500*k], (5, 5)))
	pic1 = np.transpose(np.reshape(a1t[1+500*k], (5, 5)))
	pic2 = np.transpose(np.reshape(a1t[2+500*k], (5, 5)))
	pic3 = np.transpose(np.reshape(a1t[3+500*k], (5, 5)))
	pic4 = np.transpose(np.reshape(a1t[4+500*k], (5, 5)))
	pic5 = np.transpose(np.reshape(a1t[5+500*k], (5, 5)))
	pic6 = np.transpose(np.reshape(a1t[6+500*k], (5, 5)))
	pic7 = np.transpose(np.reshape(a1t[7+500*k], (5, 5)))
	pic8 = np.transpose(np.reshape(a1t[8+500*k], (5, 5)))
	pic9 = np.transpose(np.reshape(a1t[9+500*k], (5, 5)))
	space = np.asarray([ [0 for i in range(5)] for j in range(5)])

	# Stitch these all together into one picture
	picRow = np.concatenate((pic0, space, pic1, space, pic2, space, pic3, space, pic4, space, pic5, space, pic6, space, pic7, space, pic8, space, pic9), axis = 1)
	
	emptyRow = [[0 for i in range(95)] for j in range(5)]
	
	picAll = np.concatenate((picAll, picRow, emptyRow), axis = 0)

# 'binary' for black on white, 'gray' for white on black. 
# See https://matplotlib.org/examples/color/colormaps_reference.html for more color options

imgplot = plt.imshow(picAll, cmap="binary") 
plt.show()
