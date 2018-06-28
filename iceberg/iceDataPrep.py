


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 #as cv



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
	x1, x2, TRname, TRlabel, TRangle, TRonlyAngle = DataSort(train)
	x = (x1 + x2)/2
	
	x1c = np.zeros(x.shape)
	x2c = np.zeros(x.shape)
	
	s = 75

	for i in range(25): #1604
		mat = x[i]
		mat1 = x1[i]
		mat2 = x2[i]
		fillval = np.amin(x[i])
		#print x[i][0:5, 0:5], fillval
		# Find current center
		center = np.unravel_index(np.argmax(mat), mat.shape)
		#print center
		
		if center[0] < 37:
			shift = 37 - center[0]
			for j in range(shift):
				# Delete bottom row of image
				mat1 = np.delete(mat1, s-1, 0)
				mat2 = np.delete(mat2, s-1, 0)
				# Add a row of zeros on top
				mat1 = np.vstack((np.ones((1, s))*fillval, mat1 ))
				mat2 = np.vstack((np.ones((1, s))*fillval, mat2 ))

		if center[0] > 37:
			shift = center[0] - 37
			for j in range(shift):
				# Delete top row of image
				mat1 = np.delete(mat1, 0, 0)
				mat2 = np.delete(mat2, 0, 0)
				# Add a row of zeros on bottom
				mat1 = np.vstack((mat1, np.ones((1, s))*fillval ))
				mat2 = np.vstack((mat2, np.ones((1, s))*fillval ))

		if center[1] < 37:
			shift = 37 - center[1]
			for j in range(shift):
				# Delete right column of image
				mat1 = np.delete(mat1, s-1, 1)
				mat2 = np.delete(mat2, s-1, 1)
				# Add a column of zeros on left	
				mat1 = np.hstack((np.ones((s,1))*fillval, mat1 ))	
				mat2 = np.hstack((np.ones((s,1))*fillval, mat2 ))	
		
		if center[1] > 37:
			shift = center[1] - 37
			for j in range(shift):
				# Delete left column of image
				mat1 = np.delete(mat1, 0, 1)
				mat2 = np.delete(mat2, 0, 1)
				# Add a column of zeros on right		
				mat1 = np.hstack((mat1, np.ones((s,1))*fillval ))	
				mat2 = np.hstack((mat2, np.ones((s,1))*fillval ))	

		x1c[i] = mat1
		x2c[i] = mat2
	
	return x1, x2, x1c, x2c

	
	
	
x1, x2, x1c, x2c = CenterImg()
xb1 = x1.reshape((1604, 75, 75, 1))
xb2 = x2.reshape((1604, 75, 75, 1))
xbavg = (xb1 + xb2) / 2.0
xbavg = xbavg
x = np.concatenate((xb1, xb2, xbavg ), axis=3)



x1dn = np.zeros((x1.shape))
x2dn = np.zeros((x2.shape))

for i in range(25):
	xi = Norm(x[i])*255.0
	xi = xi.astype(np.uint8)
	#xi = cv2.fromarray(xi)
	print xi.shape
	h,w,c = xi.shape
	x_rgb = cv2.CreateMat(h, w, cv2.CV_32FC3)
	
	cv2.cvtColor(xi, x_rgb, cv2.COLOR_BGR2RGB)


	dst = cv2.fastNlMeansDenoisingColored(x_rgb,None,10,10,7,21)
	cv2.imshow(dst)
	dst = np.asarray(dst.astype(float))
	print dst[0:5, 0:5, 0]
	x1dn[i] = np.ravel( dst[:,:,0])
	x2dn[i] = np.ravel( dst[:,:,1])
	

ShowSquare(x1dn, x2dn)

ShowSquare(x1c, x2c)




