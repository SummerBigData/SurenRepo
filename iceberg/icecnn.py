
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
#import ijson
# Keras stuff
from keras.models import Sequential
from keras.layers import Dense




#---------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES

# fix random seed for reproducibility
np.random.seed(7)

parser = argparse.ArgumentParser()
parser.add_argument("m", help="Number of Datapoints, up to 1604", type=int)
g = parser.parse_args()

g.f1 = 75 * 75 * 2
g.f2 = 500
#g.f3 = 100
g.f4 = 1
g.epo = 50#300
g.bsize = 300

print 'You have chosen:', g
print ' '



#----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE

def DataSort(dat):
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


def Norm(mat):
	Min = np.amin(mat)
	Max = np.amax(mat)
	nMin = 0
	nMax = 1
	return ((mat - Min) / (Max - Min)) * (nMax - nMin) + nMin

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

#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE



train = pd.read_json("data/train.json")	# 1604 x 5

'''
filename = "data/test.json"
with open(filename, 'r') as f:
    objects = ijson.items(f, 'meta.view.columns.item')
    columns = list(objects)

print columns.shape
'''

# Read out the data in the two bands, as 1604 x 75 x 75 arrays
TRb1, TRb2, TRname, TRlabel, TRangle, TRonlyAngle = DataSort(train)


# DATA PREP

linband1 = TRb1.reshape((1604, 75*75))[:g.m, :]
linband2 = TRb2.reshape((1604, 75*75))[:g.m, :]
x = np.hstack((linband1, linband2))
x = Norm(x)
y = TRlabel[:g.m]
print 'x and y', x.shape, y.shape
#print 'numIcebergs', sum(y)

# KERAS NEURAL NETWORK

# create model
model = Sequential()
model.add(Dense(g.f2, input_dim=g.f1, activation='relu'))
#model.add(Dense(g.f3, activation='relu'))
model.add(Dense(g.f4, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(x, y, epochs=g.epo, batch_size=g.bsize)


# evaluate the model
scores = model.evaluate(x, y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


model.save('models/icem'+str(g.m)+'epo'+ str(g.epo)+'bsize'+ str(g.bsize))
model.save_weights('weights/icem'+str(g.m)+'epo'+ str(g.epo)+'bsize'+ str(g.bsize))


