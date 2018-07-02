
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
import argparse
#import ijson
# Keras stuff
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
import os
import iceDataPrep



#---------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES


# fix random seed for reproducibility
np.random.seed(8)

parser = argparse.ArgumentParser()
#parser.add_argument("m", help="Number of Datapoints, up to 1604", type=int)
parser.add_argument("h", help="denoising variable for all colors", type=int)
g = parser.parse_args()
g.m = 1604
g.f1 = 75 * 75 * 2
g.f2 = 50
g.f3 = 100
g.f4 = 1

g.epo = 50#300
g.bsize = 24
saveStr = 'icem'+str(g.m)+'epo'+ str(g.epo)+'bsize'+ str(g.bsize) + 'h' + str(g.h)
print 'You have chosen:', g
print ' '



#----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE



def ShowSquare(band1, band2): 
	hspace = np.zeros((75, 5, 3))
	vspace = np.zeros((5, 5*75 + 5*6, 3))
	picAll = vspace
	for i in range(5):
		pici = hspace
		for j in range(5):
			picj = np.zeros((75, 75, 3))
			picj[:,:,0] = Norm(band1[i*5+j,:,:])
			picj[:,:,1] = Norm(band2[i*5+j,:,:])
			pici = np.hstack(( pici, picj, hspace))

		picAll = np.vstack((picAll, pici, vspace))


	imgplot = plt.imshow(picAll, cmap="binary", interpolation='none') 
	plt.show()


'''
def PrepDat(b1, b2, label):
	linband1 = b1.reshape((1604, 75*75))
	linband2 = b2.reshape((1604, 75*75))
	x = np.hstack((linband1, linband2))
	x = Norm(x)
	y = label
	return x, y
'''
def getModel():
	#Building the model
	gmodel=Sequential()
	#Conv Layer 1
	gmodel.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(75, 75, 3)))
	gmodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
	gmodel.add(Dropout(0.2))
	
	#Conv Layer 2
	gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
	gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	gmodel.add(Dropout(0.2))

	#Conv Layer 3
	gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
	gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	gmodel.add(Dropout(0.2))
	
	#Conv Layer 4
	gmodel.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
	gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	gmodel.add(Dropout(0.2))

	#Flatten the data for upcoming dense layers
	gmodel.add(Flatten())
	
	#Dense Layers
	gmodel.add(Dense(512))
	gmodel.add(Activation('relu'))
	gmodel.add(Dropout(0.2))

	#Dense Layer 2
	gmodel.add(Dense(256))
	gmodel.add(Activation('relu'))
	gmodel.add(Dropout(0.2))
	
	#Sigmoid Layer
	gmodel.add(Dense(1))
	gmodel.add(Activation('sigmoid'))

	mypotim=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	gmodel.compile(loss='binary_crossentropy',
		optimizer=mypotim,
		metrics=['accuracy'])
	gmodel.summary()
	return gmodel

def get_callbacks(filepath, patience=2):
	es = EarlyStopping('val_loss', patience=patience, mode="min")
	msave = ModelCheckpoint(filepath, save_best_only=True)
	return [es, msave]


#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE


'''
train = pd.read_json("data/train.json")	# 1604 x 5


filename = "data/test.json"
with open(filename, 'r') as f:
    objects = ijson.items(f, 'meta.view.columns.item')
    columns = list(objects)

print columns.shape


# Read out the data in the two bands, as 1604 x 75 x 75 arrays
TRb1, TRb2, TRname, TRlabel, TRangle, TRonlyAngle = DataSort(train)
'''

# DATA PREP

xtr, ytr, xte, yte = iceDataPrep.dataprep()
#xtr = iceDataPrep.denoise(xtr, g.h)
#xte = iceDataPrep.denoise(xte, g.h)
'''
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
'''


print 'x and y', xtr.shape, ytr.shape
#print 'numIcebergs', sum(y)

# KERAS NEURAL NETWORK

# create model
'''
model = Sequential()
model.add(Conv2D(filters = g.f2, kernel_size = (5,5),padding = 'Valid', 
                 activation ='relu', input_shape = (75,75,3)))
#model.add(Dense(g.f2, input_shape=(75,75,3), activation='relu'))
model.add(Flatten())
model.add(Dense(g.f3, activation='relu'))
model.add(Dense(g.f4, activation='sigmoid'))
'''
model = getModel()
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print model.summary()


file_path = 'weights/' + saveStr
callbacks = get_callbacks(filepath=file_path, patience=5)

# Fit the model
#model.fit(x, y, epochs=g.epo, batch_size=g.bsize)
'''
datagen.fit(x)
model.fit_generator(datagen.flow(xtr, ytr, batch_size=g.bsize),
	steps_per_epoch = xtr.shape[0] / (g.bsize+0.0),
	epochs=g.epo,
	verbose = 1,
	validation_data=(xte, yte),
	callbacks=callbacks)
'''

#gmodel=getModel()
model.fit(xtr, ytr,
          batch_size=24,
          epochs=50,
          verbose=1,
          validation_data=(xte, yte),
          callbacks=callbacks)


# evaluate the model

print 'Accuracy on training data:'
scores = model.evaluate(xtr, ytr)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

print 'Accuracy on testing data:'
scores = model.evaluate(xte, yte)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


model.save('models/' + saveStr )
model.save_weights('weights/' + saveStr)




