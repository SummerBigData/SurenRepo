import numpy as np
## Sam's code
#def test_convolution(X, X_conv, W, b, Z, mu):
#	# For sam's code
#	from scipy.special import expit
#    # Test the convolve function by picking 1000 random patches from the input data,
#    # preprocessing it using Z and mu, and feeding it through the SAE (using W and b)
#    #
#    # If the result is close to the convolved patch, we're good

#	patch_dim = int(np.sqrt(W.shape[1]/3.0)) # 8
#	conv_dim = X.shape[1] - patch_dim + 1 # 57

#	for i in range(1000):
#		feat_no = np.random.randint(0, W.shape[0])
#		img_no = np.random.randint(0, 7)
#		img_row = np.random.randint(0, conv_dim)
#		img_col = np.random.randint(0, conv_dim)

#		patch_x = (img_col, img_col+patch_dim)
#		patch_y = (img_row, img_row+patch_dim)

#		# Obtain a 8x8x3 patch and flatten it to length 192
#		patch = X[img_no, patch_y[0]:patch_y[1], patch_x[0]:patch_x[1], :]

#		patch = np.concatenate((patch[:,:,0].flatten(), patch[:,:,1].flatten(), patch[:,:,2].flatten())).reshape(-1, 1)
#		#patch = patch.reshape(-1, 1)

#		# Preprocess the patch
#		patch -= mu.reshape(192,1)
#		patch = Z.dot(patch) 

#		# Feed the patch through the autoencoder weights
#		# now sae_patch.shape = (400 192) . (192 1) = (400 1)
#		sae_feat = expit(W.dot(patch) + b)

#		# Compare it to the convolved patch
#		conv_feat = X_conv[:,img_no,img_row,img_col]
#		#print conv_feat.reshape(20, 20)
#		err = abs(sae_feat[feat_no, 0] - conv_feat[feat_no])
#		'''
#		import matplotlib.pyplot as plt
#		img = np.zeros((20, 42))
#		img[:,:20] = sae_feat.reshape(20, 20)
#		img[:,22:] = conv_feat.reshape(20, 20)
#		plt.imshow(img, cmap='gray')
#		plt.show()
#		'''
#		if err > 1e-9:
#			print err
#	return 'ok'

## Sam's code
#def convolve(X, W, b, Z, mu):
#    # For sam's code
#    from scipy.special import expit
#    WT, bm = ConvPrep(W, Z, b, mu)
#    WT = W.dot(Z)
#    dim = int(np.sqrt(W.shape[1]/3.0)) # 8
#    out_size = X.shape[1] - dim + 1 # 57
#    num_chan = X.shape[3] # 3
#    num_img = X.shape[0] # m - 8 for testing, 2000 for training
#    res = np.zeros(( W.shape[0],num_img, out_size, out_size)) # mx400x57x57

#    for k, x in enumerate(X):
#        for j, w in enumerate(WT):
#            conv_img = np.zeros((out_size, out_size), dtype=np.float64)
#            for i in range(num_chan):
#                xchan, wchan = x[:,:,i], w[i*dim**2 : (i+1)*dim**2].reshape(dim, dim)
#                wchan = np.flipud(np.fliplr(wchan))
#                conv_img += convolve2d(xchan, wchan, mode='valid')
#            conv_img += bm[j]
#            res[j,k,:,:] = expit(conv_img)

#    #print 'Convolution finished in %d seconds' % int(time() - start)
#    return res

d3 = np.arange(36).reshape(3,4,3)

print d3
print ' '

raveld3 = np.ravel(d3)

print raveld3.reshape(3,4,3)
