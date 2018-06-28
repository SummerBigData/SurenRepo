


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#---------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES






#----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE



def DataSort(dat):
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

def Norm(mat):
	Min = np.amin(mat)
	Max = np.amax(mat)
	nMin = 0
	nMax = 1
	return ((mat - Min) / (Max - Min)) * (nMax - nMin) + nMin

#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE


def dataprep():

	train = pd.read_json("data/train.json")	# 1604 x 5
	
	# Read out the data in the two bands, as 1604 x 75 x 75 arrays
	TRb1, TRb2, TRname, TRlabel, TRangle, TRonlyAngle = DataSort(train)

	# DATA PREP
	xb1 = TRb1.reshape((1604, 75, 75, 1))
	xb2 = TRb2.reshape((1604, 75, 75, 1))
	xbavg = (xb1 + xb2) / 2.0
	x = np.concatenate((xb1, xb2, xbavg ), axis=3)

	xtr = x[:1000,:,:,:]
	xte = x[1000:,:,:,:]
	ytr = TRlabel[:1000]
	yte = TRlabel[1000:]
	return xtr, ytr, xte, yte






