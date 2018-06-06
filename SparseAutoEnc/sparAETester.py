# Written by: 	Suren Gourapura
# Written on: 	June 6, 2018
# Purpose: 	To write a Sparce Auto-Encoder following directions from: http://deeplearning.stanford.edu/wiki/index.php/Exercise:Sparse_Autoencoder
# Goal:		Python code to calculate probabilities from W values

# Import the modules
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io
import argparse


#---------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES



parser = argparse.ArgumentParser()
parser.add_argument("m", help="Number of Datapoints to test on", type=int)
#parser.add_argument("f1", help="Number of Features (pixels) in images", type=int)
#parser.add_argument("f2", help="Number of Features in hidden layer", type=int)
parser.add_argument("lamb", help="Lambda, the overfitting knob", type=float)
parser.add_argument("beta", help="Beta, sparsity knob", type=float)
#parser.add_argument("eps", help="Bounds for theta matrix randomization, [-eps, eps]", type=float)
parser.add_argument("tolexp", help="Exponent of tolerance of minimize function, good value 10e-4, so -4", type=int)
parser.add_argument("randData", help="Use fresh, random data or use the saved data file (true or false)", type=str)
g = parser.parse_args()
g.f1 = 64
g.f2 = 25


saveStr = 'WArrs/m' + str(g.m)+ 'Tol'+str(g.tolexp)+'Lamb'+str(g.lamb)+'fone'+str(g.f1)+'ftwo'+str(g.f2)+'rand'+g.randData+'.out'



print 'You have chosen:', g
print ' '



#----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE



# Calculate the Hypothesis (layer 3) using just layer 1.
def ForwardProp(WAll, a1):
	W1, W2, b1, b2 = unLinWAll(WAll)
	# Calculate a2 (g.m x 25)
	a2 = hypothesis(W1, b1, a1)
	# Calculate and return the output from a2 and W2 (g.m x 64)
	a3 = hypothesis(W2, b2, a2)
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
# Get data. Grab the saved data
picDat = np.genfromtxt('data/rand10kSAVE.out', dtype=float)
if g.randData == 'true':
	print "Selecting rand10k data"
	picDat = np.genfromtxt('data/rand10k.out', dtype=float)
# Roll up data into matrix. Restrict it to [0,1]. Trim array to user defined size
dat = np.asarray(picDat.reshape(10000,64))
# Normalize each image
for i in range(g.m):
	dat[i] = Norm(dat[i])
a1 = dat[0:g.m, :]

# Obtain the best theta values from the text file
bestWAll = np.genfromtxt(saveStr, dtype=float)
#bestWAll = np.genfromtxt('WArrs/m10000Tol-4Lamb0.01fone64ftwo25.out', dtype=float)


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
hspaceAll = np.asarray([ [0 for i in range(53)] for j in range(5)])
picAll = hspaceAll

for i in range(10):
	# Store the pictures
	picA1 = np.reshape(np.ravel(a1[i*100]), (8,8))
	picA2 = np.reshape(np.ravel(a2[i*100]), (5,5))
	picA3 = np.reshape(np.ravel(a3[i*100]), (8,8))
#	print np.linalg.norm(a1[i*100])
#	print np.linalg.norm(a3[i*100])
	# DISPLAY PICTURES
	# To display a2 in revprop, a1, and a2 in forward prop, we design some spaces
	hspace = np.asarray([ [0 for i in range(8)] for j in range(8)])
	vspace1 = np.asarray([ [0 for i in range(5)] for j in range(1)])
	vspace2 = np.asarray([ [0 for i in range(5)] for j in range(2)])

	# We stitch the vertical spaces onto the pictures
	picA2All = np.concatenate((vspace1, picA2, vspace2), axis = 0)
	# We stitch the horizontal pictures together
	picAlli = np.concatenate((hspace, picA1, hspace, picA2All, hspace, picA3, hspace), axis = 1)
	# Finally, add this to the picAll
	picAll = np.vstack((picAll, picAlli, hspaceAll))

# Display the pictures
imgplot = plt.imshow(picAll, cmap="binary", interpolation='none') 
plt.savefig('results/a123'+'Tol'+str(g.tolexp)+'Lamb'+str(g.lamb)+'rand'+g.randData+'.png',transparent=False, format='png')
plt.show()



# We also want a picture of the activations for each node in the hidden layer
W1, W2, b1, b2 = unLinWAll(bestWAll)
W1Len = np.sum(W1**2)**(-0.5)
X = W1 / W1Len			# (25 x 64)
X = Norm(X)

picX = np.zeros((25,8,8))
for i in range(25):
	picX[i] = np.reshape(np.ravel(X[i]), (8,8))

hblack = np.asarray([ [1 for i in range(52)] for j in range(2)])
vblack = np.asarray([ [1 for i in range(2)] for j in range(8)])

picAll = hblack
for i in range(5):
	pici = np.concatenate((vblack, picX[5*i+0], vblack, picX[5*i+1], vblack, picX[5*i+2], vblack, picX[5*i+3], vblack, picX[5*i+4], vblack), axis = 1)
	picAll = np.vstack((picAll, pici, hblack))

# Display the pictures
imgplot = plt.imshow(picAll, cmap="binary", interpolation='none') 
plt.savefig('results/aHL'+'Tol'+str(g.tolexp)+'Lamb'+str(g.lamb)+'rand'+g.randData+'.png',transparent=False, format='png')
plt.show()













