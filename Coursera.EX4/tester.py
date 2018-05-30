
import numpy as np



def Linearize(a, b):
	return np.concatenate((np.ravel(a), np.ravel(b)))

def Unlinearize(vec, a1, a2, b1, b2):
	a = vec[0:a1*a2]
	b = vec[a1*a2:len(vec)]
	a = np.reshape(a, (a1, a2))
	b = np.reshape(b, (b1, b2))
	return a, b

def g(char):
	if char == 'n':		# number of data points (number of 'number' pictures)
		return 10
	if char == 'f1':	# number of features (pixels)
		return 10
	if char == 'f2':	# number of features (hidden layer)
		return 25
	if char == 'lamb':	# the 'overfitting knob'
		return 1
	if char == 'eps':	# used for generating random theta matrices
		return 0.12

def randData(xvals, yvals):
	XandY = np.hstack((xvals, yvals))
	print XandY
	np.random.shuffle(XandY)
	xVals = XandY[0:g('n'),0:g('f1')]
	yVals = XandY[0:g('n'),g('f1'):g('f1')+1]
	return xVals, yVals

def trunc(xvals, yvals, pos):	
	f = g('n') / 10			# Pick out n/10 instances for each number
	if pos == 'first':
		xVals = xvals[0:f, 0:g('f1')]	# Put the zeros in
		yVals = yvals[0:f]
		for i in range(9):
			xVals = np.append(xVals, xvals[3*(i+1) : f+3*(i+1), 0:g('f1')], axis=0)
			yVals = np.append(yVals, yvals[3*(i+1) : f+3*(i+1)])
	if pos == 'last':
		xVals = xvals[3-f:3, 0:g('f1')]	# Put the zeros in
		yVals = yvals[3-f:3]
		for i in range(9):
			xVals = np.append(xVals, xvals[3*(i+2)-f : 3*(i+2), 0:g('f1')], axis=0)
			yVals = np.append(yVals, yvals[3*(i+2)-f : 3*(i+2)])
	return xVals, yVals

yarr = np.asarray([[j] for j in range(30)])
print yarr

hypo = np.asarray([[i+8*j for i in range(10)] for j in range(30)])
print hypo


x, y = trunc(hypo, yarr, 'last')
print x
print y
