# Import the modules
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def hypothesis(theta, x1):
	
	xArr = np.array([[1]*len(x1), x1])
	thetaArr = np.asarray(theta)
	hypo = [0 for m in range(len(x1))]
	hypo = np.matmul(xArr.transpose(), thetaArr.transpose())
	return hypo

def column(matrix, i):
    return [row[i] for row in matrix]

# Obtain the x and y data values and convert them from arrays to lists
# First, we use genfromtxt to get arrays of data. The x array is 2 dimensional, so we need to split it into x1 and x2
xarr = np.genfromtxt("ex2x.dat", dtype=float)
yarr = np.genfromtxt("ex2y.dat", dtype=float)
x = xarr.tolist()
y = yarr.tolist()

# Initialize the theta and temptheta lists we will use
theta = [-3,-1]
temptheta = [0,0]

# Initialize gen as the number of generations the code runs for. Also Initialize allTheta to record all values of theta calculated
gen = 100
allTheta = [[0 for i in range(2)] for j in range(gen + 1)]
allTheta[0] = [-3,-1]

# Iterate the following for gen number of times
for g in range(gen):
    
    # Fill the temporary theta list for the y-int 
	temptheta[0] = theta[0] - (0.001) * (1.0 / len(x)) * np.sum( hypothesis(theta, x) - y )

    # Fill the temporary theta list for the slope in Land Area (x1)
	temptheta[1] = theta[1] - (0.001) * (1.0 / len(x)) * np.sum( np.matmul(hypothesis(theta, x)- y, x) )  

    # equate temptheta to theta so that the loop iterate and make progress
	for j in range(2):
        	theta[j] = temptheta[j]
		allTheta[g + 1][j] = theta[j]	# Also record the value of thetas for calculating J_Cost

# To display the result, we create a list with J values for points from the calculated thetas
J_Cost = [0 for g in range(gen + 1)] # The '+ 1' is there so that the zeroth generation is also recorded

for g in range(gen + 1):
	J_Cost[g] = (1.0 / len(x)) * 0.5 * np.sum( (hypothesis(allTheta[g], x)- y) ** 2) # Calculate and store the J values


# Plot the result using matplotlib
# The plotting code was from: https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html#scatter-plots
# Make data
JVals = [[0 for i in range(100)] for j in range(100)]
Xrange = 5
Yrange = 5
theta_0 = np.arange(-Xrange, Xrange, Xrange *2 / 100.0)
theta_1 = np.arange(-Yrange,Yrange, Yrange*2 / 100.0)

for i in range(100):
	for j in range(100):
		thetaPlot = [ theta_0[i], theta_1[j] ]
		JVals[i][j] = (1.0 / len(x)) * 0.5 * np.sum( (hypothesis(thetaPlot, x)- y) ** 2)

# For plotting, we create X, Y: These contain the mesh of all points created by combinatins of theta_0 and theta_1
X, Y = np.meshgrid(theta_0, theta_1)

# Plot the surface
fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(X, Y, JVals, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.plot(column(allTheta, 0), column(allTheta, 1), J_Cost, linewidth = 2, label='Gradient')

ax.scatter(theta[0], theta[1], J_Cost[gen], label ="Final Point")
ax.scatter(-3, -1, J_Cost[0], label ="Initial Point")
plt.xlabel('theta_0')
plt.ylabel('theta_1')
ax.legend()

# Customize the z axis.
ax.set_zlim(0, 500)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

## We also need the generations on the x-axis
#generations = [m for m in range(gen + 1)]

## Plot the result using matplotlib
#plt.plot(generations, J_Cost, 'g--') # Other color choices('o',  'r--')
#plt.xlabel('Generations/Iterations')
#plt.ylabel('J Cost')
#plt.title('J Cost vs. Generations')
#plt.legend(['Data'])
#plt.show()


