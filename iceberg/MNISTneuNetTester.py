
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
# Keras stuff
from keras.models import Sequential, load_model
from keras.layers import Dense
# REading data
import struct as st
import gzip


#---------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES



# fix random seed for reproducibility
np.random.seed(7)

parser = argparse.ArgumentParser()
parser.add_argument("m", help="Number of Datapoints, up to 1604", type=int)
parser.add_argument("epo", help="Number of Datapoints, up to 300", type=int)
parser.add_argument("bsize", help="Number of Datapoints, up to 300", type=int)
g = parser.parse_args()

g.f1 = 784
g.f2 = 800
g.f3 = 100
g.f4 = 10


print 'You have chosen:', g
print ' '

modelStr = 'models/m'+str(g.m)+'epo'+ str(g.epo)+'bsize'+ str(g.bsize)
weightStr = 'weights/m'+str(g.m)+'epo'+ str(g.epo)+'bsize'+ str(g.bsize)
g.m = 10000



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

def accuracy(model, testX, testY):      #testY is HotEncoded!!
	prediction = model.predict(testX)
	prediction = np.array(np.argmax(prediction, axis=1)) 
	testY = np.array(np.argmax(testY, axis=1))
	#print prediction[0:5], testY[0:5]
	accur = np.array([1 for (a,b) in zip(prediction,testY) if a==b ]).sum()/(g.m+0.0)
	return accur, prediction



#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE



xdat = read_idx('data/t10k-images-idx3-ubyte.gz', g.m)
ydat = read_idx('data/t10k-labels-idx1-ubyte.gz', g.m)


x = np.ravel(xdat).reshape((g.m, g.f1))/255.0
y = GenYMat(ydat)


model = load_model(modelStr)
model.load_weights(weightStr)


accuracy, prediction = accuracy(model, x, y)

print accuracy


