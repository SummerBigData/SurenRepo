# Import the modules
import numpy as np
import matplotlib.pyplot as plt
import math

# Obtain the x and y data values and convert them from arrays to lists
# First, we use genfromtxt to get arrays of data. The x array is 2 dimensional, so we need to split it into x1 and x2
xarr = np.genfromtxt("ex3x.dat", dtype=float)
yarr = np.genfromtxt("ex3y.dat", dtype=float)
y = yarr.tolist()
x1 = [0 for m in range(len(y))]
x2 = [0 for m in range(len(y))]

for m in range(len(y)):
	x1[m] = xarr[m][0].tolist()
	x2[m] = xarr[m][1].tolist()

# We want to normalize the data. Divide by the standard deviation and subtract the mean. Done in 2 loops so the mean and stdev don't change in the loop
x1Temp = [0 for m in range(len(y))]
x2Temp = [0 for m in range(len(y))]
for m in range(len(x1)):
	x1Temp[m] = (x1[m]  - np.mean(x1))/ np.std(x1)
	x2Temp[m] = (x2[m]  - np.mean(x2))/ np.std(x2)

for m in range(len(x1)):
	x1[m] = x1Temp[m]
	x2[m] = x2Temp[m]


# Initialize and fill the x and theta matricies we will use
xArr = np.transpose( np.array([[1]*len(x1), x1, x2]) )
yArr = yarr

# Calculate theta solutions. Use (x^T x)^-1 x^T y
thetaSoln0 =  np.matmul(np.transpose(xArr), xArr)		# x^T x
thetaSoln1 = np.linalg.inv(thetaSoln0)				# (x^T x)^-1 
thetaSoln2 = np.matmul( thetaSoln1, np.transpose(xArr) )	# (x^T x)^-1 x^T
thetaSoln3 = np.matmul( thetaSoln2, np.transpose(yArr) )	# (x^T x)^-1 x^T y

print(thetaSoln3)




