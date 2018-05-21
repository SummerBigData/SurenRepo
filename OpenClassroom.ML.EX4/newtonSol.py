# Import the modules
import numpy as np
import matplotlib.pyplot as plt
from math import exp

# Calculate the Hypothesis
def hypothesis(theta, xArr):
	thetaArr = np.asarray(theta)
	oldhypo = np.matmul(thetaArr, np.transpose(xArr) )
	newhypo = [0 for m in oldhypo]
	for m in range(len(oldhypo)):
		newhypo[m] = 1/(1+exp(-1*oldhypo[m]))
	return newhypo

# Calculate the Hessian
def Hessian(theta, xArr, m):
# Initialize the hessian matrix
	H = [[0 for i in range(3)] for j in range(3)]
	for i in range(m):
# Get the hypothesis out as an array
		hypoArr = np.asarray(hypothesis(theta, xArr))
# Calculate the h(x)*(1-h(x)) scalar
		hypos = hypoArr[i]* (1 - hypoArr[i])
# Now, the x*x^T matrix
		xVec = np.outer(xArr[i], xArr[i])
# Add this matrix onto H
		H = np.add(H, hypos * xVec)
	H = (1.0/m)*H
	return H

# Calculate grad J
def gradJ(theta, xArr, yArr, m):
	GradJ = [0 for i in range(3)]
# Get the hypothesis out as an array
	hypoArr = np.asarray(hypothesis(theta, xArr))
	for i in range(m):
		GradJ = np.add(GradJ, np.multiply((hypoArr[i] - yArr[i]), xArr[i]) )
	return (1.0/m)*GradJ


# Obtain the x and y data values and convert them from arrays to lists
# First, we use genfromtxt to get arrays of data. The x array is 2 dimensional, so we need to split it into x1 and x2
xarr = np.genfromtxt("ex4x.dat", dtype=float)
yarr = np.genfromtxt("ex4y.dat", dtype=float)
y = yarr.tolist()
x1 = [0 for m in range(len(y))]
x2 = [0 for m in range(len(y))]

for m in range(len(y)):
	x1[m] = xarr[m][0].tolist()
	x2[m] = xarr[m][1].tolist()

m = len(x1)
# We want to normalize the data. Divide by the standard deviation and subtract the mean. Done in 2 loops so the mean and stdev don't change in the loop
x1Temp = [0 for i in range(len(y))]
x2Temp = [0 for i in range(len(y))]
x1StdStor = 0


for i in range(m):
	x1Temp[i] = x1[i] / np.std(x1)
	x2Temp[i] = x2[i] / np.std(x2)

for i in range(m):
	x1[i] = x1Temp[i]
	x2[i] = x2Temp[i]

for i in range(m):
	x1[i] = x1[i] - np.mean(x1Temp)
	x2[i] = x2[i] - np.mean(x2Temp)

# Initialize and fill the x and theta matricies we will use
xArr = np.transpose( np.array([[1]*len(x1), x1, x2]) )
yArr = yarr

# Initialize the theta and temptheta lists we will use
theta = [1,2,3]
temptheta = [0,0,0]

# Initialize gen as the number of generations the code runs for. Also Initialize allTheta to record all values of theta calculated
gen = 50
allTheta = [[0 for i in range(3)] for j in range(gen + 1)]

# Iterate the following for gen number of times
for g in range(gen):

    # Fill the temporary theta list for the y-int 
	temptheta = np.asarray(theta) - np.matmul( np.linalg.inv(Hessian(theta, xArr, m)), gradJ(theta, xArr, yArr, m) )	

    # equate temptheta to theta so that the loop iterate and make progress
	for j in range(3):
        	theta[j] = temptheta[j]
		allTheta[g + 1][j] = theta[j]	# Also record the value of thetas for calculating J_Cost

## To display the result, we create a list with J values for points from the calculated thetas
#J_Cost = [0 for g in range(gen + 1)] # The '+ 1' is there so that the zeroth generation is also recorded

#for g in range(gen + 1):
#	J_Cost[g] = (1.0 / len(x1)) * 0.5 * np.sum( (hypothesis(allTheta[g], x1, x2)- y) ** 2) # Calculate and store the J values

# print best theta vals
print(theta)

# Make an array that stores the predicted x2 values for the given x1 values. For plotting the seperation line
detail = 100
x1Range = 5
plotx2 = [0 for i in range(detail)]
plotx1 = np.arange(-x1Range, x1Range, x1Range / (detail*0.5))

for i in range(detail):
	plotx2[i] = -1 * theta[0]/theta[2] - theta[1]* plotx1[i] /theta[2]

np.random.seed(19680801)
# Plot the result using matplotlib
colors = y
plt.scatter(x1, x2, s = 100, c = colors, alpha=0.5)
plt.plot(plotx1, plotx2, 'g--') # Other color choices('o',  'r--')
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.title('Admission Results')
plt.legend(['Data', 'Decision Boundary'])
plt.show()





