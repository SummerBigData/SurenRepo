


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



def CenterImg():
	train = pd.read_json("data/train.json")	# 1604 x 5
	
	# Read out the data in the two bands, as 1604 x 75 x 75 arrays
	TRb1, TRb2, TRname, TRlabel, TRangle, TRonlyAngle = DataSort(train)
	x = (TRb1 + TRb2)/2
	brightSpots = np.zeros((1604, 20, 20))
	for i in range(1604-1580):
		bright = np.unravel_index(np.argmax(x[i]), x[i].shape)
		print bright
	
	#print np.argwhere( np.amax(x[10]) - 0.0001 < x[10] )
	
	#ShowSquare(TRb1, TRb2)
	
	
CenterImg()








