
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import argparse
import json
# Keras stuff
import keras
from keras.models import load_model, Model
from keras.layers import Dense, Input, Lambda, Conv2D, Conv2DTranspose, Flatten, Reshape
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.losses import mse, binary_crossentropy
from keras import backend as K
import os.path # To check if a file exists
import iceDataPrep


#---------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES


# fix random seed for reproducibility
np.random.seed(8)

parser = argparse.ArgumentParser()
#parser.add_argument("m", help="Number of Datapoints, up to 1604", type=int)

g = parser.parse_args()
g.m = 1604
g.f1 = 75 * 75 * 3	# Original dimension
g.f2 = 512		# Intermediate dimension
g.f3 = 2		# Latent dimension

g.epo = 50#50
g.bsize = 128

saveStr = 'iceVAE'+str(g.epo)+'Bsize'+str(g.bsize)
print 'You have chosen:', g
print ' '



#----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE



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

def plotLatDim(encoder, dat, label):
	z_mean, _, _ = encoder.predict(dat, batch_size=g.bsize)
	plt.figure(figsize=(12, 10))
    	plt.scatter(z_mean[:, 0], z_mean[:, 1], c=label, s=4)
    	plt.colorbar()
    	plt.xlabel("z[0]")
    	plt.ylabel("z[1]")
    	plt.savefig('results/VAEhlAll.png')
    	plt.show()


def sampling(args):
    	"""Reparameterization trick by sampling for an isotropic unit Gaussian.
  	# Arguments:
        	args (tensor): mean and log of variance of Q(z|X)
    	# Returns:
      		z (tensor): sampled latent vector
    	"""

	z_mean, z_log_var = args
	batch = K.shape(z_mean)[0]
	dim = K.int_shape(z_mean)[1]
	# by default, random_normal has mean=0 and std=1.0
	epsilon = K.random_normal(shape=(batch, dim))
	return z_mean + K.exp(0.5 * z_log_var) * epsilon


# We choose a high patience so the algorthim keeps searching even after finding a maximum
def get_callbacks(filepath, patience=5):	
	es = EarlyStopping('val_loss', patience=patience, mode="min")
	msave = ModelCheckpoint(filepath, monitor='val_loss',save_best_only=True,save_weights_only=True)
	return [es, msave]

def getTest():
# Grab the unlabeled data
	json_data = open("data/test.json").read()
	dat = json.loads(json_data)
	b1, b2, name, angle  = iceDataPrep.DataSortTest(dat)
	# Reshape it
	xb1 = b1.reshape((b1.shape[0], 75, 75, 1))
	xb2 = b2.reshape((b1.shape[0], 75, 75, 1))
	xbavg = (xb1 + xb2) / 2.0
	#xbavg = np.zeros(xb1.shape)
	xte = np.concatenate((xb1, xb2, xbavg ), axis=3)
	return xte


#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE

# DATA PREP

# Grab the training (tr) and testing (te) data and labels
xtr, ytr, atr, xte, yte, ate = iceDataPrep.dataprep()
xunlab = getTest()
yunlab = np.ones((xunlab.shape[0]))*0.5 # We make them have the same label for plotting purposes
# Stitch data together into one
dat = np.concatenate((xtr, xte, xunlab), axis=0)
dat, oldmin, oldmax = iceDataPrep.Norm(dat, 0, 1)
label = np.concatenate((ytr, yte, yunlab), axis=0)

# Shuffle the data
dat, label = iceDataPrep.shuffleData(dat, label)

# Reshape data into 2D
dat = dat.reshape((dat.shape[0], g.f1))

print 'x and y', dat.shape, label.shape
#print 'numIcebergs', sum(y)



# KERAS NEURAL NETWORK


# Encoder
inputs = Input(shape=(g.f1, ), name='encoder_input')
d1 = Dense(g.f2, activation='relu')(inputs)
d2 = Dense(g.f2, activation='relu')(d1)
d3 = Dense(g.f2, activation='relu')(d2)
z_mean = Dense(g.f3, name='z_mean')(d3)
z_log_var = Dense(g.f3, name='z_log_var')(d3)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(g.f3,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()



# Decoder
latent_inputs = Input(shape=(g.f3,), name='z_sampling')
d1 = Dense(g.f2, activation='relu')(latent_inputs)
d2 = Dense(g.f2, activation='relu')(d1)
d3 = Dense(g.f2, activation='relu')(d2)
outputs = Dense(g.f1, activation='sigmoid')(d3)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# Build VAE
# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')
reconstruction_loss = mse(inputs , outputs)#* g.f1
#reconstruction_loss = np.sum(np.sum(reconstruction_loss, axis = 1), axis = 2) / (75.0**2)
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)* g.f2
print 'print 2', reconstruction_loss.shape, kl_loss.shape
vae.add_loss(vae_loss)
vae.compile(optimizer='adam', loss=None)
#vae.summary()



file_path = 'weights/' + saveStr + '.hdf5' #'{epoch:02d}-{val_loss:.2f}.hdf5'

callbacks = get_callbacks(filepath=file_path, patience=5)

# Fit VAE
vae.fit(dat,
	epochs=g.epo,
	batch_size=g.bsize,
	verbose=2,
	validation_data=(dat, None),
	callbacks=callbacks)

vae.save_weights(file_path)


plotLatDim(encoder, dat, label)



