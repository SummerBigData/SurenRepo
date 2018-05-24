# Written by: 	Suren Gourapura
# Written on: 	May 22, 2018
# Purpose: 	To solve exercise 3 on Multi-class Classification and Neural Networks in Coursera
# Goal:		To use the given theta matrices to see the classification power of a neural network

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
a1Arr = np.vstack(( np.asarray([1 for i in range(n)]) , a1))

# Calculate and store the output from a1 and theta2 (this contains the solutions)
guessAll = hypothesis(theta2, a1Arr.T)

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


# We also want to plot the a1 data as pictures, to see the hidden layer

# Generate an image. The image is inverted for some reason, so we transpose the matrix first
# We are plotting the first instance of each number in the data
a1t = a1.T
pic0 = np.transpose(np.reshape(a1t[1], (5, 5)))
pic1 = np.transpose(np.reshape(a1t[500], (5, 5)))
pic2 = np.transpose(np.reshape(a1t[1000], (5, 5)))
pic3 = np.transpose(np.reshape(a1t[1500], (5, 5)))
pic4 = np.transpose(np.reshape(a1t[2000], (5, 5)))
pic5 = np.transpose(np.reshape(a1t[2500], (5, 5)))
pic6 = np.transpose(np.reshape(a1t[3000], (5, 5)))
pic7 = np.transpose(np.reshape(a1t[3500], (5, 5)))
pic8 = np.transpose(np.reshape(a1t[4000], (5, 5)))
pic9 = np.transpose(np.reshape(a1t[4500], (5, 5)))

# Stitch these all together into one picture
picAll = np.concatenate((pic0, pic1, pic2, pic3, pic4, pic5, pic6, pic7, pic8, pic9), axis = 1)

# 'binary' for black on white, 'gray' for white on black. 
# See https://matplotlib.org/examples/color/colormaps_reference.html for more color options

imgplot = plt.imshow(picAll, cmap="binary") 
plt.show()



