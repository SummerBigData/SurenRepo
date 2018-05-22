# Written by: 	Suren Gourapura
# Written on: 	May 22, 2018
# Purpose: 	To solve exercise 3 on Multi-class Classification and Neural Networks in Coursera
# Goal:		Take the provided data and plot the numbers

# Import the modules
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Obtain the data values and convert them from arrays to lists
# First, we use genfromtxt to get arrays of data.
xarr = genfromtxt("ex3data.csv", delimiter=',', dtype=float)
n = 5000	# number of data points (number of 'number' pictures)
f = 400		# number of features (pixels)

# Generate an image. The image is inverted for some reason, so we transpose the matrix first
# We are plotting the first instance of each number in the data
pic0 = np.transpose(np.reshape(xarr[0], (20, 20)))
pic1 = np.transpose(np.reshape(xarr[500], (20, 20)))
pic2 = np.transpose(np.reshape(xarr[1000], (20, 20)))
pic3 = np.transpose(np.reshape(xarr[1500], (20, 20)))
pic4 = np.transpose(np.reshape(xarr[2000], (20, 20)))
pic5 = np.transpose(np.reshape(xarr[2500], (20, 20)))
pic6 = np.transpose(np.reshape(xarr[3000], (20, 20)))
pic7 = np.transpose(np.reshape(xarr[3500], (20, 20)))
pic8 = np.transpose(np.reshape(xarr[4000], (20, 20)))
pic9 = np.transpose(np.reshape(xarr[4500], (20, 20)))

# Stitch these all together into one picture
picAll = np.concatenate((pic0, pic1, pic2, pic3, pic4, pic5, pic6, pic7, pic8, pic9), axis = 1)

# 'binary' for black on white, 'gray' for white on black. 
# See https://matplotlib.org/examples/color/colormaps_reference.html for more color options

imgplot = plt.imshow(picAll, cmap="binary") 
plt.show()
