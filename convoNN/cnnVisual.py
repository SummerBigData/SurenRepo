# Written by: 	Suren Gourapura
# Written on: 	June 20, 2018
# Purpose: 	To write a Convolutional Neural Network
# Source:	Following directions from: http://deeplearning.stanford.edu/wiki/index.php/Exercise:Convolution_and_Pooling
# Goal:		Test the cnn.py code's results

import numpy as np
from scipy.optimize import minimize
import scipy.io
import time
import argparse
import matplotlib.pyplot as plt
import dataPrep



#---------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES



parser = argparse.ArgumentParser()
parser.add_argument("m", help="Number of images, usually 2k", type=int)
parser.add_argument("f2", help="Number of Features in hidden layer", type=int)
parser.add_argument("lamb", help="Lambda, usually 1e-4", type=float)
g = parser.parse_args()
g.f1 = 3600
#g.f2 = 36
g.f3 = 4


datStr = 'convolvedData/testm' + '3200' + 'CPRate100part' 
#datStr = 'convolvedData/m' + '2000' + 'CPRate100part'
WAllStr = 'WArrs/m' + str(g.m) + 'HL' +str(g.f2)+ 'lamb' + str(g.lamb) + '.out'
print 'You have chosen:', g
g.m = 3200

print ' '

#----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE



# Unlinearize: Take a vector, break it into two vectors, and roll it back up
def unLinWAll(vec):	
	W1 = np.asarray([vec[0				: g.f2*g.f1]])
	W2 = np.asarray([vec[g.f2*g.f1 			: g.f2*g.f1 + g.f3*g.f2]])
	b1 = np.asarray([vec[g.f2*g.f1+g.f3*g.f2	: g.f2*g.f1+g.f3*g.f2 + g.f2]])
	b2 = np.asarray([vec[g.f2*g.f1+g.f3*g.f2 + g.f2 : ]])
	return W1.reshape(g.f2, g.f1) , W2.reshape(g.f3, g.f2), b1.reshape(g.f2, 1), b2.reshape(g.f3, 1)


# Linearize: Take 4 matrices, unroll them, and stitch them together into a vector
def Lin4(a, b, c, d):
	return np.concatenate((np.ravel(a), np.ravel(b), np.ravel(c), np.ravel(d)))


# Calculate the softmax Hypothesis (for layer 2 to 3)
def softHypo(W, b, a):
	Max = np.amax(np.matmul(W, a.T) + b)	# To not blow up the np.exp
	numer = np.exp( np.matmul(W, a.T) + b - Max )	
	denom = np.asarray([np.sum(numer, axis=0)])
	return (numer/denom).T
# Calculate the logistic Hypothesis (for layer 1 to 2)
def logHypo(W, b, a):
	oldhypo = np.matmul(W, a.T) + b
	#oldhypo = np.array(oldhypo, dtype=np.float128)
	newhypo = 1.0/(1.0+np.exp(-oldhypo))	
	return newhypo.T #np.array(newhypo.T, dtype=np.float64)

# Calculate the Hypothesis (layer 3) using just layer 1.
def ForwardProp(WAll, a1):
	W1, W2, b1, b2 = unLinWAll(WAll)
	# Calculate a2 (g.m x 25)
	a2 = logHypo(W1, b1, a1)
	# Calculate and return the output from a2 and W2 (g.m x 64)
	a3 = softHypo(W2, b2, a2)
	return a2, a3, W1, W2, b1, b2

# Generate the y-matrix. This is called only once, so I use loops
def GenYMat(yvals):
	yvals = np.ravel(yvals)
	yArr = np.zeros((len(yvals), g.f3))
	for i in range(len(yvals)):
		for j in range(g.f3):
			if yvals[i] == j:
				yArr[i][j] = 1
	return yArr

# Calculate the percent correct
def PercentCorrect(guesses, daty):
	# Parse through guesses, calculate the highest guess, and store
	guessBest = np.zeros((g.m))
	for j in range(g.m):
		tempGuessAll = np.ravel(guesses[j])
		guessBest[j] = tempGuessAll.argmax()	# Record the index of the highest value

	# Now we want to see what percent of each number the code got right
	# Since the data isn't distributed evenly, we also record the frequency of each number in the dataset
	numPercent = np.zeros((guesses.shape[1]))
	numNumbers = np.zeros((guesses.shape[1])).astype(int)

	# Increment numPercent[i] for each number i it gets right.
	# Also increment the numNumbers array for whichever datapoint it was
	for i in range(g.m):
		numNumbers[ daty[i] ] += 1
		if guessBest[i] == daty[i]:
			numPercent[ daty[i] ] += 1.0

	print 'Number of data points identified correctly per number:'
	print np.array2string(numPercent, separator=',')
	print ' ' 
	numCorrect = sum(numPercent)
	print 'Number of total data points identified correctly:', numCorrect
	numPercent = numPercent / numNumbers
	print ' ' 
	print 'Percent correct per number:' 
	print np.array2string(numPercent, separator=',')
	print' '
	print'Total percent correct:', numCorrect / sum(numNumbers)
	
	return guessBest


#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE

# DATA PROCESSING

# Get the convolved and pooled images. 
print "Grabbing the convolved and pooled data..."
# These are stored in 4 files, so I stitch them together
cpdat = np.genfromtxt(datStr+'1.out', dtype=float)
for i in range(3): 		
	cpdat = np.concatenate(( cpdat, np.genfromtxt(datStr+str(i+2)+'.out', dtype=float) ))
cpdat = cpdat.reshape((400, 3200, 3, 3))[:,:g.m,:,:] 
# We need this in 2D form. g.m x 3600  (3 x 3 x 400 = 3600)
a1 = np.swapaxes(cpdat, 0, 1)
a1 = a1.reshape(g.m, 3600)	
print "Got the data"
print ' '


# Get the y labels. Labeled in set [1, 4]
y = scipy.io.loadmat('data/stlTestSubset.mat')['testLabels'] 
#y = scipy.io.loadmat('data/stlTrainSubset.mat')['trainLabels']
y = np.ravel(y[:g.m,0])-1
# Generate the y matrix. # 15298 x 10
ymat = GenYMat(y)


# Get the WAll array calculated by cnn.py
WAll = np.genfromtxt(WAllStr, dtype=float)


# FORWARD PROPAGATE AND CALCULATE PERCENTAGES
a2, a3, W1, W2, b1, b2 = ForwardProp(WAll, a1)
imgs = dataPrep.GenSubTest()	# 3.2k x 64 x 64 x 3 

guessBest = PercentCorrect(a3, y)

# Extract the wrong guess positions
wrongGuess = np.zeros((g.m)).astype(int)
for i in range(g.m):
	if guessBest[i] != y[i]:
		wrongGuess[i] = 1

# Grab relavant data: Which images were wrong, what they were guessed as, and what they actually were
numWrong = np.sum(wrongGuess)
imgsWrong = np.zeros((numWrong, 64, 64, 3))
predGuess = np.zeros((numWrong))
actGuess = np.zeros((numWrong))

ind = 0
for i in range(numWrong):
	if wrongGuess[i] == 1:
		imgsWrong[ind] = imgs[i]
		predGuess[ind] = guessBest[i]
		actGuess[ind] = y[i]
		ind += 1

	
hspace = np.ones((64, 5, 3))
vspace = np.ones((25, 5*64+5*2, 3))
picAll = vspace
for i in range(5):
	pici = hspace
	for j in range(5):
		pici = np.hstack((pici, imgsWrong[i*5+j]))
	pici = np.hstack((pici, hspace))
	picAll = np.vstack((picAll, vspace, pici))

picAll = np.vstack((picAll, vspace))

imgplot = plt.imshow(picAll, cmap="binary", interpolation='none') 
plt.show()


## SHOW SOME IMAGES
#print y[0:25] # cat

#hspace = np.zeros((64, 5, 3))
#vspace = np.zeros((5, 5*64+5*2, 3))
#picAll = vspace
#for i in range(5):
#	pici = hspace
#	for j in range(5):
#		pici = np.hstack((pici, imgs[i*5+j]))
#	pici = np.hstack((pici, hspace))
#	picAll = np.vstack((picAll, pici))

#picAll = np.vstack((picAll, vspace))

#imgplot = plt.imshow(picAll, cmap="binary", interpolation='none') 
#plt.show()





## PLOT HIDDEN LAYER
## Generate an image. The image is inverted for some reason, so we transpose the matrix first
## We are plotting the first ten instances of each number in the data
#a1 = hypothesis(bestTheta1, xArr)
#a1t = a1.T

#f = g('n')/10
#s = int(np.sqrt(g('f2')))
#picAll = [ [0 for i in range(19*s)] for j in range(5)]

#for k in range(10):
#	pic0 = a1t[0+f*k].reshape((s,s))
#	pic1 = a1t[1+f*k].reshape((s,s))
#	pic2 = a1t[2+f*k].reshape((s,s))
#	pic3 = a1t[3+f*k].reshape((s,s))
#	pic4 = a1t[4+f*k].reshape((s,s))
#	pic5 = a1t[5+f*k].reshape((s,s))
#	pic6 = a1t[6+f*k].reshape((s,s))
#	pic7 = a1t[7+f*k].reshape((s,s))
#	pic8 = a1t[8+f*k].reshape((s,s))
#	pic9 = a1t[9+f*k].reshape((s,s))
#	space = np.asarray([ [0 for i in range(s)] for j in range(s)])

#	# Stitch these all together into one picture
#	picRow = np.concatenate((pic0, space, pic1, space, pic2, space, pic3, space, pic4, space, pic5, space, pic6, space, pic7, space, pic8, space, pic9), axis = 1)
#	
#	emptyRow = [[0 for i in range(19*s)] for j in range(s)]
#	
#	picAll = np.concatenate((picAll, picRow, emptyRow), axis = 0)

## 'binary' for black on white, 'gray' for white on black. 
## See https://matplotlib.org/examples/color/colormaps_reference.html for more color options

#imgplot = plt.imshow(picAll, cmap="binary", interpolation='none') 
#plt.show()





