# Written by: 	Suren Gourapura
# Written on: 	May 31, 2018
# Purpose: 	To solve exercise 4 on Multi-class Classification and Neural Networks in Coursera
# Goal:		Get pictures of previous layers

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
		return 500 #FIX
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


def Normalize(arr):
#	print "Smallest Val: ", np.amin(arr)
#	print "Largest Val: ", np.amax(arr)
	arr = arr + np.abs(np.amin(arr)) + 0.01
	arr =  arr / np.amax(arr) - 0.001
#	print "new Smallest Val: ", np.amin(arr)
#	print "new Largest Val: ", np.amax(arr)
	return arr

def RevProp(aLplus1, theta):
	# Apply the inverse sigmoid function
	faLplus1 = np.log( -aLplus1 / (aLplus1 - 1.0))
	# Calculate the inverse theta matrix. 
	#Note that pinv is an approximation of the inverse matrix for non-square matrices, not exact
	invtheta = np.linalg.pinv(theta)
	# Calculate aL
	return np.dot(invtheta, faLplus1)

def ForProp(aL, theta):
	# Add a 1
	aL = np.hstack(([[1]], [np.ravel(aL)]))	
	# Calculate layer L plus 1
	return hypothesis(theta, aL)







#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE



# PREPARING DATA
# Obtain the best theta values from the text file
bestThetas = np.genfromtxt('neuralThetas4500.1.out', dtype=float)

# Seperate and reform the theta matrices
bestTheta1, bestTheta2 = UnLin(bestThetas, g('f2'), g('f1')+1, 10, g('f2')+1)



# REVERSE PROPAGATE
# Prepare the output layer. 'numb' is the number it will generate
numb = 3
#a3 = np.zeros(10)
#for  i in range(10):
#	if i == numb:
#		a3[i] = 0.999	# We can't make these 1 and 0 since the inverse sigmoid doesn't like those values
#	else:
#		a3[i] = 0.001
a3 = np.array([  3.72642259e-03,  2.58019391e-03,  9.34899206e-03,  8.75294055e-01,   1.83352066e-03,  2.92327749e-02,  6.99613542e-05,  1.43447250e-03,   7.18752968e-03,  1.02925693e-02])



# Reverse propagate to layer 2
a2 = RevProp(a3, bestTheta2)

# Trim a2 to 25 units
a2 = a2[1:26]

## Since the Inverse Sigmoid function doesn't like values outside of (0, 1), we normalize the data.
a2 = Normalize(a2)

# Store this picture. Note that we are storing the normalized picture to better see the dark areas
picRa2 = np.transpose(np.reshape(Normalize(a2), (5,5)))

# Reverse propagate to layer 1
a1 = RevProp(a2, bestTheta1)
	
# Trim a1 to 400 units
a1 = a1[1:401]

# Store this picture
picRa1 = np.transpose(np.reshape(Normalize(a1), (20,20)))



# FORWARD PROPAGATE
# Propagate to a2
a2 = ForProp(a1, bestTheta1)

# Store this picture
picFa2 = np.transpose(np.reshape(np.ravel(Normalize(a2)), (5,5)))

# Propagate to a3. These are the probabilities
a3 = ForProp(a2, bestTheta2)
np.set_printoptions(suppress=True)

print(np.array2string(np.ravel(a3), separator=','))


# DISPLAY PICTURES
# To display a2 in revprop, a1, and a2 in forward prop, we design some spaces
hspace = np.asarray([ [0 for i in range(20)] for j in range(20)])
vspace1 = np.asarray([ [0 for i in range(5)] for j in range(7)])
vspace2 = np.asarray([ [0 for i in range(5)] for j in range(8)])

# We stitch the vertical spaces onto the pictures
picRa2 = np.concatenate((vspace1, picRa2, vspace2), axis = 0)
picFa2 = np.concatenate((vspace1, picFa2, vspace2), axis = 0)

# We stitch the pictures together
picAll = np.concatenate((hspace, picRa2, hspace, picRa1, hspace, picFa2, hspace), axis = 1)

# Display the pictures
imgplot = plt.imshow(picAll, cmap="binary", interpolation='none') 
plt.savefig('results/RevPropReAvg'+str(numb)+'.png',transparent=True, format='png')
plt.show()






