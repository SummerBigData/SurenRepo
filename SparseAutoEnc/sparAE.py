# Written by: 	Suren Gourapura
# Written on: 	June 5, 2018
# Purpose: 	To write a Sparce Auto-Encoder following directions from: http://deeplearning.stanford.edu/wiki/index.php/Exercise:Sparse_Autoencoder
# Goal:		Python code to calculate W vals


import numpy as np
from math import log
from scipy.optimize import minimize
import scipy.io
import time
import argparse
import matplotlib.pyplot as plt
#from scipy.optimize import check_grad
from random import randint
import randpicGen


#---------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES



parser = argparse.ArgumentParser()
parser.add_argument("m", help="Number of Datapoints", type=int)
parser.add_argument("f1", help="Number of Features (pixels) in images", type=int)
parser.add_argument("f2", help="Number of Features in hidden layer", type=int)
parser.add_argument("lamb", help="Lambda, the overfitting knob", type=float)
#parser.add_argument("eps", help="Bounds for theta matrix randomization, [-eps, eps]", type=float)
parser.add_argument("tolexp", help="Exponent of tolerance of minimize function, good value 10e-4, so -4", type=int)
parser.add_argument("randData", help="Use fresh, random data or use the saved data file (true or false)", type=str)

g = parser.parse_args()
saveStr = 'WArrs/m' + str(g.m)+ 'Tol'+str(g.tolexp)+'Lamb'+str(g.lamb)+'fone'+str(g.f1)+'ftwo'+str(g.f2)+'.out'
gStep = 0
eps = 0.12

print 'You have chosen:', g
print 'Will be saved in: ', saveStr
print ' '


#----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE




def saveTheta(theta):
	np.savetxt(saveStr, theta, delimiter=',')



#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE

# To see how long the code runs for, we start a timestamp
totStart = time.time()

# Get data. Grab the saved data
picDat = np.genfromtxt('data/rand10kSAVED.out', dtype=float)
# If user wants fresh data, run randpicGen.py and rewrite picDat with this data
if g.randData == 'true':
	randpicGen.GenDat()
	picDat = np.genfromtxt('data/rand10k.out', dtype=float)

dat = np.asarray(picDat.reshape(10000,8,8))















# Stop the timestamp and print out the total time
totend = time.time()
print'sparAE.py took ', totend - totStart, 'seconds to run'



