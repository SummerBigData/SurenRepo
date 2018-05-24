# Written by: 	Suren Gourapura
# Written on: 	May 22, 2018
# Purpose: 	To solve exercise 3 on Multi-class Classification and Neural Networks in Coursera
# Goal:		?

# Import the modules
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
from math import exp, log
from scipy.optimize import minimize
import scipy.io


#----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE


# Calculate the Hypothesis
def hypothesis(thetaArr, xArr):
	oldhypo = np.matmul(thetaArr, np.transpose(xArr) )
	newhypo = 1/(1+np.exp(-oldhypo))
	return newhypo



# Take out one column (column i) from a matrix
def column(matrix, i):
    return [row[i] for row in matrix]


#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE
	

# Obtain the data values and convert them from arrays to lists
data = scipy.io.loadmat('ex3data1.mat')
weights = scipy.io.loadmat('ex3weights.mat')
xvals = data['X']		# 5000 x 400 
yvals = data['y']
theta1 = weights['Theta1']	# 25 x 401 matrix that takes x to a_1
theta2 = weights['Theta2']	# 10 x 26 matrix that takes a_1 to output (y)
n = 5000	# number of data points (number of 'number' pictures)
f = 400		# number of features (pixels)

# Form the correct x array, with xArr[0:5000, 0:1] being a column of 1's
xArr = np.hstack(( np.asarray([[1] for i in range(n)]) , xvals))

# Calculate a1 using theta1 and the x data
a1 = hypothesis(theta1, xArr)

# Add a row of 1's on top of the a1 matrix
a1 = np.vstack(( np.asarray([1 for i in range(n)]) , a1))

# Calculate and store the output from a1 and theta2 (this contains the solutions)
guessAll = hypothesis(theta2, a1.T)

# We need to parse through this 10 x 5000, find the highest values, and record them in a 5000 1D array
guessBest = np.asarray([0 for i in range(n)])
for j in range(n):
	tempGuessAll = np.ravel(column(guessAll,j))
	guessBest[j] = tempGuessAll.argmax()	# Record the index of the highest value

# Print some sample sections where the guesses should transition, from 0-1, 1-2, 2-3, 8-9
print guessBest[480:510], guessBest[980:1010], guessBest[1490:1510], guessBest[4490:4510]

# Now we want to see what percent of each number the code got right
numPercent = np.array([0.0 for i in range(10)])

# The matrices report 0 as 9, 1 as 0, 2 as 1, 3 as 2, etc. To fix this, we hardcode a translation when calculating the percentages

for i in range(n):
	if yvals[i] == 10 and guessBest[i] == 9:
		numPercent[0] = numPercent[0] + 1.0
	elif guessBest[i]+1 == yvals[i]:
		numPercent[ yvals[i] ] = numPercent[ yvals[i] ] + 1.0


numPercent = numPercent * (1/ 500.0)

print numPercent



