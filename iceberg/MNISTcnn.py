
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
# Keras stuff
from keras.models import Sequential
from keras.layers import Dense
# REading data
import struct as st
import gzip


#---------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES



# fix random seed for reproducibility
np.random.seed(7)

parser = argparse.ArgumentParser()
parser.add_argument("m", help="Number of Datapoints, up to 1604", type=int)
g = parser.parse_args()

g.f1 = 784
g.f2 = 500
g.f3 = 100
g.f4 = 10
g.epo = 300
g.bsize = 300

print 'You have chosen:', g
print ' '



#----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE



# Read the MNIST dataset
def read_idx(filename, n=None):
	with gzip.open(filename) as f:
		zero, dtype, dims = st.unpack('>HBB', f.read(4))
		shape = tuple(st.unpack('>I', f.read(4))[0] for d in range(dims))
		arr = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
		if not n is None:
			arr = arr[:n]
		return arr

def GenYMat(yvals):
	yvals = np.ravel(yvals)
	yArr = np.zeros((g.m, 10))
	for i in range(len(yvals)):
		for j in range(10):
			if yvals[i] == j or (yvals[i] == 10 and j == 0):
				yArr[i][j] = 1
	return yArr



#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE



xdat = read_idx('data/train-images-idx3-ubyte.gz', g.m)
ydat = read_idx('data/train-labels-idx1-ubyte.gz', g.m)
	
x = np.ravel(xdat).reshape((g.m, g.f1))/255.0
y = GenYMat(ydat)
print 'x and y', x.shape, y.shape

# KERAS NEURAL NETWORK

# create model
model = Sequential()
model.add(Dense(g.f2, input_dim=g.f1, activation='relu'))
model.add(Dense(g.f3, activation='relu'))
model.add(Dense(g.f4, activation='softmax'))

# Compile model
model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
print model.summary()

# Fit the model
model.fit(x, y, epochs=g.epo, batch_size=g.bsize)


# evaluate the model
scores = model.evaluate(x, y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


model.save('models/m'+str(g.m)+'epo'+ str(g.epo)+'bsize'+ str(g.bsize))
model.save_weights('weights/m'+str(g.m)+'epo'+ str(g.epo)+'bsize'+ str(g.bsize))
