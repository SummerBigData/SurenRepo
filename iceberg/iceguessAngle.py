import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
import argparse
#import ijson
# Keras stuff
#import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
import os
import iceDataPrep



#---------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES


# fix random seed for reproducibility
np.random.seed(7)

parser = argparse.ArgumentParser()
#parser.add_argument("m", help="Number of Datapoints, up to 1604", type=int)
#parser.add_argument("h", help="denoising variable for all colors", type=int)
g = parser.parse_args()
g.m = 1604
g.f1 = 75 * 75 * 2
g.f2 = 50
g.f3 = 100
g.f4 = 1

g.epo = 60#300
g.bsize = 24
saveStr = 'iceGuessAnglem'+str(g.m)+'epo'+ str(g.epo)+'bsize'+ str(g.bsize) #+ 'h' + str(g.h)
print 'You have chosen:', g
print ' '



#----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE


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
	gmodel.add(Activation('linear'))

	mypotim=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	gmodel.compile(loss='mean_squared_error', optimizer=mypotim, metrics=['mae'])
	gmodel.summary()
	return gmodel

def get_callbacks(filepath, patience=2):
	es = EarlyStopping('val_loss', patience=patience, mode="min")
	msave = ModelCheckpoint(filepath, save_best_only=True)
	return [es, msave]


#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE



xtr, atr, xte, ate = iceDataPrep.dataprepAngle()
print xtr.shape, atr.shape, xte.shape, ate.shape
print ate.dtype, ate[0:5]
print xte.dtype, 

model = getModel()
# Compile model
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])



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
model.fit(xtr, atr,
	batch_size=g.bsize,
	epochs=g.epo,
	verbose=1,
	validation_data=(xte, ate),
	callbacks=callbacks)


# evaluate the model

print 'Accuracy on training data:'
scores = model.evaluate(xtr, atr)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

print 'Accuracy on testing data:'
scores = model.evaluate(xte, ate)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print ate[0:30]
print model.predict(xte[0:30])

model.save('models/' + saveStr )
model.save_weights('weights/' + saveStr)



