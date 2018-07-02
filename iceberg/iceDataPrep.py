


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

def ShowSquare(x, rows, cols): 
	hspace = np.zeros((75, 5, 3))
	vspace = np.zeros((5, cols*75 + 5*(cols+1), 3))
	picAll = vspace
	for i in range(rows):
		pici = hspace
		for j in range(cols):
			# Put together the three bands
			c1, a, b = Norm(x[i*cols+j,:,:,0], 0, 1)
			c2, a, b = Norm(x[i*cols+j,:,:,1], 0, 1)
			c3, a, b = Norm(x[i*cols+j,:,:,2], 0, 1)
			c1 = c1.reshape((75,75,1))
			c2 = c2.reshape((75,75,1))
			c3 = c3.reshape((75,75,1))
			picj = np.concatenate((c1, c2, c3), axis=2)
			# Include the image in the row
			pici = np.hstack(( pici, picj, hspace))

		picAll = np.vstack((picAll, pici, vspace))

	imgplot = plt.imshow(picAll, cmap="binary", interpolation='none') 
	plt.show()


def Norm(mat, nMin, nMax):
	# Calculate the old min, max and convert to float values
	Min = np.amin(mat).astype(float)
	Max = np.amax(mat).astype(float)
	nMin = nMin+0.0
	nMax = nMax+0.0
	# Calculate the new normalized matrix
	normMat = ((mat - Min) / (Max - Min)) * (nMax - nMin) + nMin
	return normMat, Min, Max

#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE


def dataprep():

	train = pd.read_json("data/train.json")	# 1604 x 5
	
	# Read out the data in the two bands, as 1604 x 75 x 75 arrays
	TRb1, TRb2, TRname, TRlabel, TRangle, TRonlyAngle = DataSort(train)

	# DATA PREP
	xb1 = TRb1.reshape((1604, 75, 75, 1))
	xb2 = TRb2.reshape((1604, 75, 75, 1))
	xbavg = (xb1 + xb2) / 2.0
	#xbavg = np.zeros(xb1.shape)
	x = np.concatenate((xb1, xb2, xbavg ), axis=3)

	xtr = x[:1000,:,:,:]
	xte = x[1000:,:,:,:]
	ytr = TRlabel[:1000]
	yte = TRlabel[1000:]
	return xtr, ytr, xte, yte


'''
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

'''


def denoise(x,H):
	print 'Denoising the images'
	
	'''
	xdn = np.zeros((x.shape))
	for i in range(x.shape[0]): # 1604
		# Force values between 0 and 255, but remember their original scaling
		xi, oldMinMax = NormColors(x[i], 0, 255.0)
		
		xi = xi.astype(np.uint8)
		
		x_rgb = cv2.cvtColor(xi, cv2.COLOR_BGR2RGB)
		dst = cv2.fastNlMeansDenoisingColored(x_rgb,h,hcolor,7,21)	# 10,10,7,21
	
		xdn[i] = np.asarray(dst.astype(float))
		
		# To preserve the original scaling, we renormalize with old min/max
		xdn[i] = unNormColors(xdn[i], oldMinMax)
	'''
	
	xdn = np.zeros((x.shape))
	xdn[:,:,:,2] = x[:,:,:,2]
	for i in range(x.shape[0]): # 1604
		# Force values between 0 and 255, but remember their original scaling
		xi, oldMinMax = NormColors(x[i], 0, 255.0)
		
		xi = xi.astype(np.uint8)
		for j in range(2):
			#x_grey = cv2.cvtColor(xi[:, :, j], cv2.CV_RGB2GRAY)
			dst = cv2.fastNlMeansDenoising(src = xi[:, :, j] ,h=H,templateWindowSize = 7,searchWindowSize = 21)	# 10,10,7,21
	
			xdn[i, :, :, j] = np.asarray(dst.astype(float))
		
		# To preserve the original scaling, we renormalize with old min/max
		xdn[i] = unNormColors(xdn[i], oldMinMax)
	print 'done denoising'
	return xdn

def NormColors(mat, nMin, nMax): # accepts mat.shape -> (75, 75, 3)
	oldMinMax = np.zeros((3, 2)) # 3 colors, min and max
	normMat = np.zeros(mat.shape)
	for i in range(3):
		normMat[:,:,i],oldMinMax[i,0],oldMinMax[i,1] = Norm(mat[:, :, i], nMin, nMax)
	return normMat, oldMinMax

def unNormColors(mat, oldMinMax):
	unNormMat = np.zeros(mat.shape)
	for i in range(3):
		unNormMat[:,:,i],a,b = Norm(mat[:, :, i], oldMinMax[i,0], oldMinMax[i,1])
	return unNormMat



'''	
xtr, ytr, xte, yte = dataprep()

xtr = xtr[0:10, :, :, :]
#xtr[:,:,:,0] = np.zeros((1,75,75))
#xtr[:,:,:,2] = np.zeros((1,75,75))

x = np.zeros((100, 75, 75, 3))
for i in range(10):
	for j in range(10):
		x[i*10+j] = denoise(xtr[i:i+1], j*10)

#x = np.concatenate(( xtr, xtrdn), axis = 0)

ShowSquare(x, 10, 10)

'''


'''
xtr = xtr[2:3, :, :, :]
xtr[:,:,:,0] = np.zeros((1,75,75))
xtr[:,:,:,2] = np.zeros((1,75,75))

x = np.zeros((16, 75, 75, 3))
for i in range(4):
	for j in range(4):
		x[i*4+j] = denoise(xtr[0:1],i*100, j*3)

#x = np.concatenate(( xtr, xtrdn), axis = 0)

ShowSquare(x, 4, 4)

'''
