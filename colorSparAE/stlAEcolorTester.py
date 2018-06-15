# Written by: 	Suren Gourapura
# Written on: 	June 6, 2018
# Purpose: 	To write a Sparce Auto-Encoder following directions from: http://deeplearning.stanford.edu/wiki/index.php/Exercise:Sparse_Autoencoder
# Goal:		Python code to calculate probabilities from W values

# Import the modules
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import scipy.io
import argparse
import dataPrepColor



#---------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES



parser = argparse.ArgumentParser()
parser.add_argument("m", help="Number of Datapoints, usually 29404", type=int)
#parser.add_argument("f1", help="Number of Features (pixels) in images", type=int)
#parser.add_argument("f2", help="Number of Features in hidden layer", type=int)
parser.add_argument("lamb", help="Lambda, the overfitting knob", type=float)
parser.add_argument("beta", help="Beta, sparsity knob", type=float)
#parser.add_argument("eps", help="Bounds for theta matrix randomization, [-eps, eps]", type=float)
parser.add_argument("tolexp", help="Exponent of tolerance of minimize function, good value 10e-4, so -4", type=int)

g = parser.parse_args()

#g.m = 0 # Will be adjusted later
gStep = 0
g.eps = 0.12
g.f1 = 192
g.f2 = 400
g.rho = 0.05
#g.beta = 3
saveStr = 'WArrs/m' + str(g.m)+ 'Tol'+str(g.tolexp)+'Lamb'+str(g.lamb)+'beta'+str(g.beta)+'.out'

print 'You have chosen:', g
print ' '



#----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE



# Calculate the Hypothesis for a1 -> a2
def hypoA12(W, b, dat):
	oldhypo = np.matmul(W, dat.T) + b
	oldhypo = np.array(oldhypo, dtype=np.float128)	# Helps prevent overflow errors
	newhypo = 1.0/(1.0+np.exp(-oldhypo))	
	return np.array(newhypo.T, dtype=np.float64)

# Calculate the Hypothesis for a2 -> a3
def hypoA23(W, b, dat):
	oldhypo = np.matmul(W, dat.T) + b
	return oldhypo.T

# Calculate the Hypothesis (layer 3) using just layer 1.
def ForwardProp(WAll, a1):
	W1, W2, b1, b2 = unLinWAll(WAll)
	# Calculate a2 (g.m x 25)
	a2 = hypoA12(W1, b1, a1)
	# Calculate and return the output from a2 and W2 (g.m x 64)
	a3 = hypoA23(W2, b2, a2)
	return a2, a3

# Unlinearize: Take a vector, break it into two vectors, and roll it back up
def unLinWAll(vec):	
	W1 = np.asarray([vec[0			: g.f2*g.f1]])
	W2 = np.asarray([vec[g.f2*g.f1 		: g.f2*g.f1*2]])
	b1 = np.asarray([vec[g.f2*g.f1*2 	: g.f2*g.f1*2 + g.f2]])
	b2 = np.asarray([vec[ g.f2*g.f1*2 + g.f2 : g.f2*g.f1*2 + g.f2 + g.f1]])
	return W1.reshape(g.f2, g.f1) , W2.reshape(g.f1, g.f2), b1.reshape(g.f2, 1), b2.reshape(g.f1, 1)

def PlotImg(mat):
	imgplot = plt.imshow(mat, cmap="binary", interpolation='none') 
	plt.show()

# Calculate the Hypothesis (for layer l to l+1)
def hypothesis(W, b, dat):
	oldhypo = np.matmul(W, dat.T) + b
	oldhypo = np.array(oldhypo, dtype=np.float128)	# Helps prevent overflow errors
	newhypo = 1.0/(1.0+np.exp(-oldhypo))	
	return np.array(newhypo.T, dtype=np.float64)

def Norm(mat):
	Min = np.amin(mat)
	Max = np.amax(mat)
	nMin = 0
	nMax = 1
	return ((mat - Min) / (Max - Min)) * (nMax - nMin) + nMin

#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE



# DATA PROCESSING


# Get data. Call the data by acccessing the function in dataPrepColor
dat = dataPrepColor.GenDat()	# 100k x 64 x 3
dat = dat[:g.m, :, :]
whitenedDat, ZCAmat = dataPrepColor.zcaWhite(dat)

# Another way, pull the matrix from the saved data
#ZCAmat = np.genfromtxt('data/m100.0kZCA.out', dtype=float).reshape(192,192)

# Reshape and normalize the data
a1 = Norm(whitenedDat.reshape(g.m, g.f1))
#print np.amax(dat), np.amin(dat)



# Obtain the best theta values from the text file
bestWAll = np.genfromtxt(saveStr, dtype=float)


# FORWARD PROPAGATE AND CALCULATE BEST GUESSES
# Feed the best W and b vals into forward propagation
a2, a3 = ForwardProp(bestWAll, a1)

for i in range(g.m):
	a2[i] = Norm(a2[i])
	a3[i] = Norm(a3[i])


# PROBABILITIES
prob = np.zeros((g.m, g.f1))
for i in range(g.m):
	prob[i] = np.abs(a1[i]-a3[i])

print 'The average seperation between a1 and a3 is (Note: 0-1, where 0 is close)', np.mean(prob)


# SHOW IMAGES

s = int((g.f1/3)**(0.5))

# Pic Facts
v = 8	# How many pictures in a coumn
linsp = 2

vspace = np.zeros((linsp, linsp*3+s*2, 3))
hspace = np.zeros((s, linsp, 3))
picAll = vspace

for i in range(10):
	# Store the pictures
	picA1 = np.reshape(np.ravel(a1[i]), (s,s, 3))
	#picA2 = np.reshape(np.ravel(a2[i]), (sa2,sa2,3))
	picA3 = np.reshape(np.ravel(a3[i]), (s,s, 3))
	
	# We stitch the horizontal pictures together
	picAlli = np.concatenate((hspace, picA1, hspace, picA3, hspace), axis = 1)
	# Finally, add this to the picAll
	picAll = np.vstack((picAll, picAlli, vspace))

# Display the pictures
imgplot = plt.imshow(picAll, cmap="binary", interpolation='none') 
plt.savefig('results/a123m' + str(g.m)+ 'Tol'+str(g.tolexp)+'Lamb'+str(g.lamb)+'beta'+str(g.beta)+'.png',transparent=False, format='png')
plt.show()



# We also want a picture of the activations for each node in the hidden layer
W1, W2, b1, b2 = unLinWAll(bestWAll)
W1Len = np.sum(W1**2)**(-0.5)
X = W1 / W1Len	
X = np.matmul(X, ZCAmat)		
X = Norm(X)

picX = np.zeros((g.f2,s,s, 3))
for i in range(g.f2):
	picX[i] = np.reshape(np.ravel(X[i]), (s,s, 3))


# Pic Facts
h = 20 	# How many pictures in a row
v = 20	# How many pictures in a coumn
linsp = 2

# Note, all at 1 == white. All at 0 == black. 
hblack = np.zeros((linsp, s*h+linsp*(v+1), 3))
vblack = np.zeros((s, linsp, 3))

picAll = hblack
for i in range(v):
	pici = vblack
	for j in range(h):
		pici = np.hstack(( pici, picX[i*h+j], vblack))

	picAll = np.vstack((picAll, pici, hblack))

# Display the pictures
imgplot = plt.imshow(picAll, cmap="binary", interpolation='none') 
plt.savefig('results/aHLm' + str(g.m)+ 'Tol'+str(g.tolexp)+'Lamb'+str(g.lamb)+'beta'+str(g.beta)+'.png',transparent=False, format='png')
plt.show()













