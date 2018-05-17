# Import the modules
import numpy as np
import matplotlib.pyplot as plt
import math

def hypothesis(theta, x1, x2):
	
	xArr = np.array([[1]*len(x1), x1, x2])
	thetaArr = np.asarray(theta)
	hypo = [0 for m in range(len(x1))]
	hypo = np.matmul(xArr.transpose(), thetaArr.transpose())
	return hypo


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

# We want to normalize the data. Divide by the means
x1mean = 0
x2mean = 0
for m in range(len(x1)):
	x1mean = x1[m] + x1mean
	x2mean = x2[m] + x2mean

x1mean = x1mean / len(x1)
x2mean = x2mean / len(x1)

for m in range(len(x1)):
	x1[m] = x1[m] / x1mean
	x2[m] = x2[m] / x2mean

# Initialize the theta and temptheta lists we will use
theta = [0,0,0]
temptheta = [0,0,0]

# Initialize gen as the number of generations the code runs for. Also Initialize allTheta to record all values of theta calculated
gen = 50
allTheta = [[0 for i in range(3)] for j in range(gen + 1)]

# Iterate the following for gen number of times
for g in range(gen):
    
    # Fill the temporary theta list for the y-int 
	temptheta[0] = theta[0] - (0.01) * (1.0 / len(x1)) * np.sum( hypothesis(theta, x1, x2) - y )

    # Fill the temporary theta list for the slope in Land Area (x1)
	temptheta[1] = theta[1] - (0.01) * (1.0 / len(x1)) * np.sum( np.matmul(hypothesis(theta, x1, x2)- y, x1) )  

    # Fill the temporary theta list for the slope in # Bedrooms (x2)
	temptheta[2] = theta[2] - (0.01) * (1.0 / len(x1)) * np.sum( np.matmul(hypothesis(theta, x1, x2)- y, x2) )

    # equate temptheta to theta so that the loop iterate and make progress
	for j in range(3):
        	theta[j] = temptheta[j]
		allTheta[g + 1][j] = theta[j]	# Also record the value of thetas for calculating J_Cost

# To display the result, we create a list with J values for points from the calculated thetas
J_Cost = [0 for g in range(gen + 1)] # The '+ 1' is there so that the zeroth generation is also recorded

for g in range(gen + 1):
	J_Cost[g] = (1.0 / len(x1)) * 0.5 * np.sum( (hypothesis(allTheta[g], x1, x2)- y) ** 2) # Calculate and store the J values

# We also need the generations on the x-axis
generations = [m for m in range(gen + 1)]

# For plotting all data together, we output the raw J_Cost data as well
print(J_Cost)

# Plot the result using matplotlib
plt.plot(generations, J_Cost, 'g--') # Other color choices('o',  'r--')
plt.xlabel('Generations/Iterations')
plt.ylabel('J Cost')
plt.title('J Cost vs. Generations')
plt.legend(['Data'])
plt.show()
