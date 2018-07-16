
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import argparse
# Keras stuff
import json
from keras.models import Sequential, load_model
from keras.layers import Dense
import iceDataPrep

#---------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES

# fix random seed for reproducibility
np.random.seed(7)

parser = argparse.ArgumentParser()
#parser.add_argument("m", help="Number of Datapoints, up to 1604", type=int)
#parser.add_argument("epo", help="Number of Datapoints, up to 300", type=int)
#parser.add_argument("bsize", help="Number of Datapoints, up to 300", type=int)
#parser.add_argument("datType", help="'Test' data or 'Train' data?", type=str)
parser.add_argument("h", help="denoising variable for all colors", type=int)
parser.add_argument("trimsize", help="how many pixels do we trim for data augmentation (even #)?", type=int)
g = parser.parse_args()
g.m = 1604
g.f1 = 75 * 75 * 2
g.f2 = 500
#g.f3 = 100
g.f4 = 1
g.epo = 70
g.bsize = 100
#if g.datType == 'Test':
#	g.testm = 8424

print 'You have chosen:', g
print ' '

#modelStr = 'models/icem'+str(g.m)+'epo'+ str(g.epo)+'bsize'+ str(g.bsize)
weightStr = 'iceEpo'+str(g.epo)+'Bsize'+str(g.bsize)+'h'+str(g.h)+'Trimsize'+str(g.trimsize)



#----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE

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
	for i in range(testX.shape[0]):
		if testY[i] == 0 and prediction[i] < 0.5:
			gotright += 1
		elif testY[i] == 1 and prediction[i] > 0.5:
			gotright += 1
	accur = gotright / (testX.shape[0] + 0.0)
	return accur, prediction

def binaryprediction(model, testX):
	prediction = model.predict(testX)
	binaryPred = np.zeros((prediction.shape[0])).astype(int)
	for i in range(testX.shape[0]):
		if prediction[i] < 0.5:
			binaryPred[i] = 0
		else:
			binaryPred[i] = 1
	return binaryPred


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

# DATA PREP

#test = pd.read_json("data/train.json")	# ? x 5

#xtr, ytr, atr, xte, yte, ate = iceDataPrep.dataprep()


json_data = open("data/test.json").read()
dat = json.loads(json_data)

b1, b2, name, angle  = iceDataPrep.DataSortTest(dat)

xb1 = b1.reshape((b1.shape[0], 75, 75, 1))
xb2 = b2.reshape((b1.shape[0], 75, 75, 1))
xbavg = (xb1 + xb2) / 2.0
#xbavg = np.zeros(xb1.shape)
xte = np.concatenate((xb1, xb2, xbavg ), axis=3)





# Load the model and weights and make the predictions
model = load_model('models/iceModel'+str(75 - g.trimsize) )
model.load_weights('weights/'+weightStr + '.hdf5')

prediction = model.predict(xte)
binaryPred = binaryprediction(model, xte)

submission = pd.DataFrame()
submission['id']= name
submission['is_iceberg']= prediction.reshape((prediction.shape[0]))
submission.to_csv('submits/sub7-16dn'+str(g.h)+'tr'+str(g.trimsize)+'.csv', index=False)






