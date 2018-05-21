# Import the modules
import numpy as np
import matplotlib.pyplot as plt
import math

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

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
	x1Temp[m] = x1[m] / np.std(x1)
	x2Temp[m] = x2[m] / np.std(x2)

for m in range(len(x1)):
	x1[m] = x1Temp[m]
	x2[m] = x2Temp[m]

for m in range(len(x1)):
	x1[m] = x1[m] - np.mean(x1Temp)
	x2[m] = x2[m] - np.mean(x2Temp)

thetaMe = [ 340412.65957447 , 109447.79646964 ,  -6578.35485416]
thetaSam = [89597.9095428, 139.21067402, -8738.01911233]

bestSolnMe = [0 for m in range(len(y))]
bestSolnSam = [0 for m in range(len(y))]

for m in range(len(y)):
	bestSolnMe[m] = thetaMe[0] + x1[m]* thetaMe[1] + x2[m]* thetaMe[2] 
	bestSolnSam[m] = thetaSam[0] + x1[m]*thetaSam[1] + x2[m]*thetaSam[2]

# Plot the result using matplotlib
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.scatter(x1, x2, y, 'g')
ax.plot(x1, x2, bestSolnMe, 'r--',linewidth = 2)
ax.set_xlabel('x1: Living Area')
ax.set_ylabel('x2: # of Bedrooms')
ax.set_zlabel('Cost')
ax.set_title('Data and Best Fit')
ax.legend(['Data', 'Best Fit'])
plt.show()




## 2D Plots:
## Plot the result using matplotlib
#plt.plot(x1, y, 'o', x1, bestSolnMe, 'r--',) # Other color choices('o', 'g--', 'r--')
#plt.xlabel('Living Area')
#plt.ylabel('Cost')
#plt.title('x1 Data and Best Fit')
#plt.legend(['Data', 'Best Fit'])
#plt.show()

## Plot the result using matplotlib
#plt.plot(x2, y, 'o',  x2, bestSolnMe, 'r--') # Other color choices('o', 'g--', 'r--')
#plt.xlabel('# Bedrooms')
#plt.ylabel('Cost')
#plt.title('x2 Data and Best Fit')
#plt.legend(['Data', 'Best Fit'])
#plt.show()
