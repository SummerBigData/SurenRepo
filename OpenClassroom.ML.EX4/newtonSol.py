# Written by: 	Suren Gourapura
# Written on: 	May 21, 2018
# Purpose: 	To solve exercise 4 on Logistic Regression in OpenClassroom
# Goal:		Take the provided data and calculate the line that best seperates the data. Plot the line on the data
# 		and plot the cost function vs. number of iterations. Calculate the best theta values and what percent 
# 		chance a student who scores x1 = 20, x2 = 80 have on passing the final.

# Import the modules
import numpy as np
import matplotlib.pyplot as plt
from math import exp, log

# Calculate the Hypothesis
def hypothesis(theta, xArr):
	thetaArr = np.asarray(theta)
	oldhypo = np.matmul(thetaArr, np.transpose(xArr) )
	newhypo = [0 for i in oldhypo]
	for i in range(len(oldhypo)):
		newhypo[i] = 1/(1+exp(-1*oldhypo[i]))
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

# We want to normalize the data. Divide by the standard deviation and subtract the mean. Done in 2 loops so the mean and stdev don't change in the loop. We record these so we can undo the effects later
x1Temp = [0 for i in range(len(y))]
x2Temp = [0 for i in range(len(y))]
x1Std  = np.std(x1)
x2Std  = np.std(x2)
x1Mean = np.mean(x1)
x2Mean = np.mean(x2)

for i in range(m):
	x1Temp[i] = (x1[i] - x1Mean )/ x1Std
	x2Temp[i] = (x2[i] - x2Mean )/ x2Std

for i in range(m):
	x1[i] = x1Temp[i]
	x2[i] = x2Temp[i]

# Initialize and fill the x and theta matricies we will use
xArr = np.transpose( np.array([[1]*len(x1), x1, x2]) )
yArr = yarr

# Initialize the theta and temptheta lists we will use
theta = [1,1,1]
temptheta = [0,0,0]

# Initialize gen as the number of generations the code runs for. Also Initialize allTheta to record all values of theta calculated
gen = 1
allTheta = [[0 for i in range(3)] for j in range(gen + 1)]

# Iterate the following for gen number of times
for g in range(gen):

    # Fill the temporary theta list for the y-int 
	temptheta = np.asarray(theta) - np.matmul( np.linalg.inv(Hessian(theta, xArr, m)), gradJ(theta, xArr, yArr, m) )	

    # equate temptheta to theta so that the loop iterate and make progress
	for j in range(3):
        	theta[j] = temptheta[j]
		allTheta[g + 1][j] = theta[j]	# Also record the value of thetas for calculating J_Cost

# For plotting the seperation line, make an array that stores the predicted x2 values for the given x1 values. 
detail = 100
x1Range = 5
plotx2 = [0 for i in range(detail)]
plotx1 = np.arange(-x1Range, x1Range, x1Range / (detail*0.5))

for i in range(detail):
	plotx2[i] = -1 * theta[0]/theta[2] - theta[1]* plotx1[i] /theta[2]

# Now we fix the scaling
for i in range(detail):
	plotx1[i] = plotx1[i]*x1Std + x1Mean
	plotx2[i] = plotx2[i]*x2Std + x2Mean
for i in range(m):
	x1[i] = x1[i]*x1Std + x1Mean
	x2[i] = x2[i]*x2Std + x2Mean

# Calculate the actual, unscaled thetas
actTheta = [0,0,0]
actTheta[0] = theta[0] - theta[1]*x1Mean/x1Std - theta[2]*x2Mean/x2Std
actTheta[1] = theta[1]/x1Std
actTheta[2] = theta[2]/x2Std
print('Best Theta Value')
print(actTheta)
print(allTheta[0])
# For plotting the progress per generation, make the cost array and calculate it's values
J = [0 for i in range(gen+1)]
generations = [i for i in range(gen+1)]

for i in range(gen+1):
	for j in range(m):
		hypo = hypothesis(allTheta[i], xArr)
		J[i] = J[i] + (1.0/m)*(  -1*y[j]*log(hypo[j]) - (1-y[j])*log(1 - hypo[j])  )
# Calculate the probability that a student with x1 = 20, x2 = 80 will not be admitted
print('Probability that [x1 = 20: x2 = 80] will pass the final')
print(1 - 1/(1+exp(-actTheta[0] - actTheta[1]*20 - actTheta[2]*80 )) )

# Plot the result using matplotlib
#colors = y
plt.scatter(x1, x2, s = 100, c = y, alpha=0.5)
plt.plot(plotx1, plotx2, 'g--') # Other color choices('o',  'r--')
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.title('Admission Results')
plt.legend(['Decision Boundary', 'Data'])
plt.show()

# Plot the fitting function's performance versus generations
plt.plot(generations, J, 'g--') # Other color choices('o',  'r--')
plt.xlabel('Generations')
plt.ylabel('J Cost')
plt.title("Newton's Method ")
plt.show()



