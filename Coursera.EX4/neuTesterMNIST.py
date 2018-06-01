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
import struct as st
import gzip

#----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE


# These are the global constants used in the code
def g(char):
	if char == 'n':		# number of data points (number of 'number' pictures)
		return 59500 	# CHANGE THIS TO ADJUST TRAINING SET SIZE
	if char == 'f1':	# number of features (pixels)
		return 784
	if char == 'f2':	# number of features (hidden layer)
		return 36
	if char == 'lamb':	# the 'overfitting knob'
		return 1
	if char == 'eps':	# used for generating random theta matrices
		return 0.12


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


# Take the xvals matrix and extract the first or last instances of each number so that the total is now g('n') training samples
def trunc(xvals, yvals, pos):	
	f = g('n') / 10					# Pick out n/10 instances for each number
	if pos == 'first':				# If the user wants the first 'f' values
		xVals = xvals[0:f, 0:g('f1')]		# Put the zeros in
		yVals = yvals[0:f]
		for i in range(9):			# Put in the rest
			xVals = np.append(xVals, xvals[500*(i+1) : f+500*(i+1), 0:g('f1')], axis=0)
			yVals = np.append(yVals, yvals[500*(i+1) : f+500*(i+1)])
	if pos == 'last':				# If the user wants the last 'f' values
		xVals = xvals[500-f:500, 0:g('f1')]	# Put the zeros in
		yVals = yvals[500-f:500]
		for i in range(9):			# Put in the rest
			xVals = np.append(xVals, xvals[500*(i+2)-f : 500*(i+2), 0:g('f1')], axis=0)
			yVals = np.append(yVals, yvals[500*(i+2)-f : 500*(i+2)])
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



# PREPARING DATA
# Obtain the data values and convert them from arrays to lists
datx = read_idx('data/train-images-idx3-ubyte.gz', g('n'))
daty = read_idx('data/train-labels-idx1-ubyte.gz', g('n'))

datx = np.ravel(datx).reshape((g('n'), g('f1')))

# Reorder the data randomly
datx, daty = randData(datx, daty)

# Form the correct x and y arrays, with a column of 1's in the xArr
xArr = np.hstack(( np.asarray([[1] for i in range(g('n'))]) , datx))	# g('n') x g('f1')	

# Obtain the best theta values from the text file
bestThetas = np.genfromtxt('neuralThetas50MNIST.out', dtype=float)

# Seperate and reform the theta matrices
bestTheta1, bestTheta2 = UnLin(bestThetas, g('f2'), g('f1')+1, 10, g('f2')+1)



# FORWARD PROPAGATE AND CALCULATE BEST GUESSES
# Calculate the best guess given the best theta values
guesses = FullHypo(bestTheta1, bestTheta2, xArr)	#10 x g('n')

# We need to parse through this 10 x 5000, find the highest values, and record them in a 5000 1D array
guessBest = np.asarray([0 for i in range(g('n'))])

for j in range(g('n')):
	tempGuessAll = np.ravel(column(guesses,j))
	guessBest[j] = tempGuessAll.argmax()	# Record the index of the highest value

# Print some sample sections where the guesses should transition, from 0-1, 1-2, 2-3, 8-9
# If testing on truncated data, this might not work without modifications
print "Here are some sections of the guessBest vector"
f = g('n')/10
print guessBest[f-10:f+10], guessBest[2*f-10:2*f+10], guessBest[3*f-10:3*f+10], guessBest[9*f-10:9*f+10]
print ' '


# CALCULATE PERCENT CORRECT
# Now we want to see what percent of each number the code got right
# Since the data isn't distributed evenly, we also record the frequency of each number in the dataset
numPercent = np.array([0.0 for i in range(10)])
numNumbers = np.array([0 for i in range(10)])

# Increment numPercent[i] for each number i it gets right.
# Also increment the numNumbers array for whichever datapoint it was
for i in range(g('n')):
	numNumbers[ daty[i] ] += 1
	if guessBest[i] == daty[i]:
		numPercent[ daty[i] ] += 1.0

print 'Number of data points identified correctly per number:'
print numPercent
print ' '
print 'Number of total data points identified correctly:', sum(numPercent)
numPercent = numPercent / numNumbers
print ' '
print 'Percent correct per number:'
print numPercent
print ' '
print 'Total percent correct:', np.mean(numPercent)



# PLOT HIDDEN LAYER
# Generate an image. The image is inverted for some reason, so we transpose the matrix first
# We are plotting the first ten instances of each number in the data
a1 = hypothesis(bestTheta1, xArr)
a1t = a1.T

f = g('n')/10
s = int(np.sqrt(g('f2')))
picAll = [ [0 for i in range(19*s)] for j in range(5)]

for k in range(10):
	pic0 = a1t[0+f*k].reshape((s,s))
	pic1 = a1t[1+f*k].reshape((s,s))
	pic2 = a1t[2+f*k].reshape((s,s))
	pic3 = a1t[3+f*k].reshape((s,s))
	pic4 = a1t[4+f*k].reshape((s,s))
	pic5 = a1t[5+f*k].reshape((s,s))
	pic6 = a1t[6+f*k].reshape((s,s))
	pic7 = a1t[7+f*k].reshape((s,s))
	pic8 = a1t[8+f*k].reshape((s,s))
	pic9 = a1t[9+f*k].reshape((s,s))
	space = np.asarray([ [0 for i in range(s)] for j in range(s)])

	# Stitch these all together into one picture
	picRow = np.concatenate((pic0, space, pic1, space, pic2, space, pic3, space, pic4, space, pic5, space, pic6, space, pic7, space, pic8, space, pic9), axis = 1)
	
	emptyRow = [[0 for i in range(19*s)] for j in range(s)]
	
	picAll = np.concatenate((picAll, picRow, emptyRow), axis = 0)

# 'binary' for black on white, 'gray' for white on black. 
# See https://matplotlib.org/examples/color/colormaps_reference.html for more color options

imgplot = plt.imshow(picAll, cmap="binary") 
plt.show()
