
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
# Keras stuff
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
# REading data
import struct as st
import gzip


#---------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES



# fix random seed for reproducibility
np.random.seed(7)

parser = argparse.ArgumentParser()
parser.add_argument("m", help="Number of Datapoints, up to 60k", type=int)
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
	
x = np.ravel(xdat).reshape((g.m, 28, 28, 1))/255.0
y = GenYMat(ydat)
print 'x and y', x.shape, y.shape

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


# KERAS NEURAL NETWORK

# create model
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

# Compile model
model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
print model.summary()

# Fit the model
datagen.fit(x)
model.fit_generator(datagen.flow(x, y, batch_size=g.bsize),
                    steps_per_epoch=len(x) / (g.bsize+0.0), epochs=g.epo)


# evaluate the model
scores = model.evaluate(x, y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


#model.save('models/m'+str(g.m)+'epo'+ str(g.epo)+'bsize'+ str(g.bsize))
#model.save_weights('weights/m'+str(g.m)+'epo'+ str(g.epo)+'bsize'+ str(g.bsize))


model.save('models//yassinem'+str(g.m)+'epo'+ str(g.epo)+'bsize'+ str(g.bsize))
model.save_weights('weights//yassinem'+str(g.m)+'epo'+ str(g.epo)+'bsize'+ str(g.bsize))








