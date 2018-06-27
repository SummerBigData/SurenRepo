
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
parser.add_argument("testm", help="Number of Datapoints, up to ?", type=int)
parser.add_argument("epo", help="Number of Datapoints, up to 300", type=int)
parser.add_argument("bsize", help="Number of Datapoints, up to 300", type=int)
g = parser.parse_args()

g.f1 = 75 * 75 * 2
g.f2 = 500
#g.f3 = 100
g.f4 = 1



print 'You have chosen:', g
print ' '

modelStr = 'models/icem'+str(g.m)+'epo'+ str(g.epo)+'bsize'+ str(g.bsize)
weightStr = 'weights/icem'+str(g.m)+'epo'+ str(g.epo)+'bsize'+ str(g.bsize)
g.m = 10000


#----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE



def DataSort(dat):
	dat = np.asarray(dat)
	print dat.shape
	print dat[0:5]
	band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in dat["band_1"]])
	band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in dat["band_2"]])

	# Read the name, label, and inclination angle as (1604,) arrays
	name = np.array(train['id'])
	label = np.array(train['is_iceberg'])	# 0 or 1
	angle = np.array(train['inc_angle'])	# angle in degrees
	print sum(label)
	# Create an array with all the 'na' angles replaced with the average angle
	# The average angle is 39.2687, the minimum is 24.7546, the maximum is 45.9375
	onlyAngle = np.zeros((1604))

	#onlyAngle = np.zeros((1604-133))		# 133 objects have no angle
	ind = 0
	for i in range(1604):
		if angle[i] == 'na':
			onlyAngle[i] = 39.2687	
		else:
			onlyAngle[i] = angle[i]
	return band1, band2, name, label, angle, onlyAngle


def accuracy(model, testX, testY):      #testY is HotEncoded!!
	prediction = model.predict(testX)
	prediction = np.array(np.argmax(prediction, axis=1)) 
	testY = np.array(np.argmax(testY, axis=1))
	#print prediction[0:5], testY[0:5]
	accur = np.array([1 for (a,b) in zip(prediction,testY) if a==b ]).sum()/(g.m+0.0)
	return accur, prediction

def Norm(mat):
	Min = np.amin(mat)
	Max = np.amax(mat)
	nMin = 0
	nMax = 1
	return ((mat - Min) / (Max - Min)) * (nMax - nMin) + nMin



#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE


print 'reading data'

#test = pd.read_json("data/test.json")	# ? x 5

json_data=open("data/test.json").read()

test = json.loads(json_data)
'''
filename = "data/test.json"
with open(filename, 'r') as f:
    objects = ijson.items(f, 'meta.view.columns.item')
    columns = list(objects)

print columns.shape
'''

print 'got data'
TEb1, TEb2, TEname, TElabel, TEangle, TEonlyAngle = DataSort(test)

print 'band 1 shape' ,TEb1.shape


linband1 = TEb1.reshape((1604, 75*75))[:g.testm, :]
linband2 = TEb2.reshape((1604, 75*75))[:g.testm, :]
x = np.hstack((linband1, linband2))
x = Norm(x)
y = TElabel[:g.testm]





model = load_model(modelStr)
model.load_weights(weightStr)

ccuracy, prediction = accuracy(model, x, y)

print accuracy




