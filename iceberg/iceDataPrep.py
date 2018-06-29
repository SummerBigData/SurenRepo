


import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
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

def ShowSquare(x1, x2, xavg): 
	hspace = np.zeros((75, 5, 3))
	vspace = np.zeros((5, 5*75 + 5*6, 3))
	picAll = vspace
	for i in range(5):
		pici = hspace
		for j in range(5):
			# Put together the three bands
			picj = np.zeros((75, 75, 3))
			picj[:,:,0] = x1[i*5+j,:,:]
			picj[:,:,1] = x2[i*5+j,:,:]
			picj[:,:,2] = xavg[i*5+j,:,:]
			# Include the image in the row
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



def CenterImgSpot():
	train = pd.read_json("data/train.json")	# 1604 x 5
	
	# Read out the data in the two bands, as 1604 x 75 x 75 arrays
	x1, x2, TRname, TRlabel, TRangle, TRonlyAngle = DataSort(train)
	x = (x1 + x2)/2
	
	x1c = np.zeros(x.shape)
	x2c = np.zeros(x.shape)
	
	s = 75
	maxshift = np.zeros((4)) #FIX

	for i in range(1604): #1604
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
			if shift == 36:
				print 'here', i
			if shift > maxshift[0]:		# FIX
				maxshift[0] = shift	# FIX

			for j in range(shift):
				# Delete bottom row of image
				mat1 = np.delete(mat1, s-1, 0)
				mat2 = np.delete(mat2, s-1, 0)
				# Add a row of zeros on top
				mat1 = np.vstack((np.ones((1, s))*fillval, mat1 ))
				mat2 = np.vstack((np.ones((1, s))*fillval, mat2 ))

		if center[0] > 37:
			shift = center[0] - 37

			if shift > maxshift[1]:		# FIX
				maxshift[1] = shift	# FIX
		
			for j in range(shift):
				# Delete top row of image
				mat1 = np.delete(mat1, 0, 0)
				mat2 = np.delete(mat2, 0, 0)
				# Add a row of zeros on bottom
				mat1 = np.vstack((mat1, np.ones((1, s))*fillval ))
				mat2 = np.vstack((mat2, np.ones((1, s))*fillval ))

		if center[1] < 37:
			shift = 37 - center[1]

			if shift > maxshift[2]:		# FIX
				maxshift[2] = shift	# FIX

			for j in range(shift):
				# Delete right column of image
				mat1 = np.delete(mat1, s-1, 1)
				mat2 = np.delete(mat2, s-1, 1)
				# Add a column of zeros on left	
				mat1 = np.hstack((np.ones((s,1))*fillval, mat1 ))	
				mat2 = np.hstack((np.ones((s,1))*fillval, mat2 ))	
		
		if center[1] > 37:
			shift = center[1] - 37

			if shift > maxshift[3]:		# FIX
				maxshift[3] = shift	# FIX

			for j in range(shift):
				# Delete left column of image
				mat1 = np.delete(mat1, 0, 1)
				mat2 = np.delete(mat2, 0, 1)
				# Add a column of zeros on right		
				mat1 = np.hstack((mat1, np.ones((s,1))*fillval ))	
				mat2 = np.hstack((mat2, np.ones((s,1))*fillval ))	

		x1c[i] = mat1
		x2c[i] = mat2
	print 'maximum shifts', maxshift
	return x1, x2, x1c, x2c





def CenterImgWeight():
	train = pd.read_json("data/train.json")	# 1604 x 5
	
	# Read out the data in the two bands, as 1604 x 75 x 75 arrays
	x1, x2, TRname, TRlabel, TRangle, TRonlyAngle = DataSort(train)
	x = (x1 + x2)/2
	
	x1c = np.zeros(x.shape)
	x2c = np.zeros(x.shape)
	
	s = 75
	maxshift = np.zeros((4)) #FIX

	for i in range(25): #1604
		mat = 1.0/x[i]
		mat1 = x1[i]
		mat2 = x2[i]
		fillval = np.amin(x[i])
		#print x[i][0:5, 0:5], fillval
		# Find current center
		center = np.zeros((2)).astype(int)
		# Cropping the edges out of the equation
		ravg = np.sum(mat[5:70, 5:70], axis = 0)
		ravg /= np.sum(ravg)
		cavg = np.sum(mat[5:70, 5:70], axis = 1)
		cavg /= np.sum(cavg)
		print ravg, cavg
		index = np.arange(65) + 5.0
		
		center0 = np.sum(ravg * index)
		center1 = np.sum(cavg * index)
		center[0] = int( round(center0, 0))
		center[1] = int( round(center1, 0))
		print i, center
		#print center
		
		if center[0] < 37:
			shift = 37 - center[0]
			if shift == 36:
				print 'here', i
			if shift > maxshift[0]:		# FIX
				maxshift[0] = shift	# FIX

			for j in range(shift):
				# Delete bottom row of image
				mat1 = np.delete(mat1, s-1, 0)
				mat2 = np.delete(mat2, s-1, 0)
				# Add a row of zeros on top
				mat1 = np.vstack((np.ones((1, s))*fillval, mat1 ))
				mat2 = np.vstack((np.ones((1, s))*fillval, mat2 ))

		if center[0] > 37:
			shift = center[0] - 37

			if shift > maxshift[1]:		# FIX
				maxshift[1] = shift	# FIX
		
			for j in range(shift):
				# Delete top row of image
				mat1 = np.delete(mat1, 0, 0)
				mat2 = np.delete(mat2, 0, 0)
				# Add a row of zeros on bottom
				mat1 = np.vstack((mat1, np.ones((1, s))*fillval ))
				mat2 = np.vstack((mat2, np.ones((1, s))*fillval ))

		if center[1] < 37:
			shift = 37 - center[1]

			if shift > maxshift[2]:		# FIX
				maxshift[2] = shift	# FIX

			for j in range(shift):
				# Delete right column of image
				mat1 = np.delete(mat1, s-1, 1)
				mat2 = np.delete(mat2, s-1, 1)
				# Add a column of zeros on left	
				mat1 = np.hstack((np.ones((s,1))*fillval, mat1 ))	
				mat2 = np.hstack((np.ones((s,1))*fillval, mat2 ))	
		
		if center[1] > 37:
			shift = center[1] - 37

			if shift > maxshift[3]:		# FIX
				maxshift[3] = shift	# FIX

			for j in range(shift):
				# Delete left column of image
				mat1 = np.delete(mat1, 0, 1)
				mat2 = np.delete(mat2, 0, 1)
				# Add a column of zeros on right		
				mat1 = np.hstack((mat1, np.ones((s,1))*fillval ))	
				mat2 = np.hstack((mat2, np.ones((s,1))*fillval ))	

		x1c[i] = mat1
		x2c[i] = mat2
	print 'maximum shifts', maxshift
	return x1, x2, x1c, x2c




def denoise(x, h, hcolor):
	print 'Denoising the images'
	'''
	xb1 = x1.reshape((x1.shape[0], 75, 75, 1))
	xb2 = x2.reshape((x1.shape[0], 75, 75, 1))
	xbavg = (xb1 + xb2) / 2.0
	xbavg = xbavg
	x = np.concatenate((xb1, xb2, xbavg ), axis=3)

	x1dn = np.zeros((x1.shape))
	x2dn = np.zeros((x2.shape))
	xavgdn = np.zeros((x1.shape))
	'''
	xdn = np.zeros((x.shape))
	for i in range(1604): # 1604
		xi = Norm(x[i])*255.0
		xi = xi.astype(np.uint8)
	
		x_rgb = cv2.cvtColor(xi, cv2.COLOR_BGR2RGB)
		dst = cv2.fastNlMeansDenoisingColored(x_rgb,h,hcolor,7,21)	# 10,10,7,21
	
		xdn[i] = np.asarray(dst.astype(float))
		'''
		x1dn[i] = dst[:,:,0]
		x2dn[i] = dst[:,:,1]
		xavgdn[i] = dst[:,:,2]
		'''
	print 'done denoising'
	return xdn


	
#xtr, ytr, xte, yte = dataprep()
#x1, x2, x1c, x2c = CenterImgWeight()
#xtr = denoise(xtr, 5, 5)

#x1dn, x2dn, xavgdn = denoise(x1, x2)
#xdn = denoise(xtr, h, hcolor)

#ShowSquare(x1dn, x2dn,xavgdn)
# 291, 718, 1568
'''
x1pic = Norm(x1c[291]).reshape(( 75, 75, 1))
x2pic = Norm(x2c[291]).reshape((75, 75, 1))
xavg = np.zeros(( 75, 75, 1))

x = np.concatenate((x1pic, x2pic, xavg  ) , axis = 2)

imgplot = plt.imshow(x, cmap="binary", interpolation='none') 
plt.show()
'''
#ShowSquare(xdn[:,:,:,0], xdn[:,:,:,1], xdn[:,:,:,2])


