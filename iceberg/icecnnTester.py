
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
# Keras stuff
import json
from keras.models import Sequential, load_model
from keras.layers import Dense


#---------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES

# fix random seed for reproducibility
np.random.seed(7)

parser = argparse.ArgumentParser()
parser.add_argument("m", help="Number of Datapoints, up to 1604", type=int)
parser.add_argument("epo", help="Number of Datapoints, up to 300", type=int)
parser.add_argument("bsize", help="Number of Datapoints, up to 300", type=int)
#parser.add_argument("datType", help="'Test' data or 'Train' data?", type=str)
g = parser.parse_args()

g.f1 = 75 * 75 * 2
g.f2 = 500
#g.f3 = 100
g.f4 = 1

#if g.datType == 'Test':
#	g.testm = 8424

g.testm = 1604

print 'You have chosen:', g
print ' '

modelStr = 'models/icem'+str(g.m)+'epo'+ str(g.epo)+'bsize'+ str(g.bsize)
weightStr = 'weights/icem'+str(g.m)+'epo'+ str(g.epo)+'bsize'+ str(g.bsize)



#----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE



def DataSortTest(dat):
	'''
	band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in dat["band_1"]])
	band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in dat["band_2"]])
	'''
	print len(dat)
	band1 = np.zeros((len(dat), 5625))
	band2 = np.zeros((len(dat), 5625))
	name = np.zeros(( len(dat) )).astype(str)
	angle = np.zeros(( len(dat) ))

	for i in range(len(dat)):

		band1[i] = np.array(dat[i]['band_1'])
		band2[i] = np.array(dat[i]['band_2'])
	
	# Read the name, label, and inclination angle as (5625,) arrays
		name[i] = np.array(dat[i]['id'])
	#label = np.array(dat['is_iceberg'])	# 0 or 1
		angle[i] = np.array(dat[i]['inc_angle'])	# angle in degrees

	return band1, band2, name, angle


def DataSortTrain(dat):
	band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in dat["band_1"]])
	band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in dat["band_2"]])

	# Read the name, label, and inclination angle as (1604,) arrays
	name = np.array(dat['id'])
	label = np.array(dat['is_iceberg'])	# 0 or 1
	angle = np.array(dat['inc_angle'])	# angle in degrees
	# Create an array with all the 'na' angles replaced with the average angle
	# The average angle is 39.2687, the minimum is 24.7546, the maximum is 45.9375
	onlyAngle = np.zeros((1604))
	ind = 0
	for i in range(1604):
		if angle[i] == 'na':
			onlyAngle[i] = 39.2687	
		else:
			onlyAngle[i] = angle[i]

	return band1, band2, name, label, angle, onlyAngle

def PrepDat(TEb1, TEb2, TElabel):
	linband1 = TEb1.reshape((g.testm, 75*75))
	linband2 = TEb2.reshape((g.testm, 75*75))
	x = np.hstack((linband1, linband2))
	x = Norm(x)
	y = TElabel
	return x, y


def binaryaccuracy(model, testX, testY):   
	prediction = model.predict(testX)
	gotright = 0
	for i in range(g.testm):
		if prediction[i] == testY[i]:
			gotright += 1
	accur = gotright / (g.testm + 0.0)
	return accur, prediction

def Norm(mat):
	Min = np.amin(mat)
	Max = np.amax(mat)
	nMin = 0
	nMax = 1
	return ((mat - Min) / (Max - Min)) * (nMax - nMin) + nMin

def GenYMat(yvals):
	yvals = np.ravel(yvals)
	yArr = np.zeros((g.m, 10))
	for i in range(len(yvals)):
		for j in range(10):
			if yvals[i] == j or (yvals[i] == 10 and j == 0):
				yArr[i][j] = 1
	return yArr

#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE


print 'reading data'

#test = pd.read_json("data/train.json")	# ? x 5
'''
if g.datType == 'Test':
	json_data = open("data/test.json").read()
	global dat
	dat = json.loads(json_data)
'''

dat = pd.read_json("data/train.json")

print 'got data'


#TEb1, TEb2, TEname, TEangle = DataSortTest(dat)

TEb1, TEb2, TEname, TElabel, TEangle, TEonlyAngle= DataSortTrain(dat)


x, y = PrepDat(TEb1, TEb2, TElabel)


#y = GenYMat(y)


print x.shape, y.shape



model = load_model(modelStr)
model.load_weights(weightStr)

accuracy, prediction = binaryaccuracy(model, x, y)

print accuracy




