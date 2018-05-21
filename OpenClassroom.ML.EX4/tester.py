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



theta = [1,1,1]
xArr1 = [[1,1,5], [1,2,6], [1,3,7], [1,4,8]]
xArr2 = [[1, 55.5,   69.5],[1, 41,   81.5],[1, 53.5,   86],[1, 46,   84]]
yArr1 = [1.0,1.0,0.0,1.0]

m = 4
#print(xArr)
#print(Hessian(theta, xArr1, m))
#print(Hessian(theta, xArr2, m))
#print(hypothesis(theta,  xArr) )
print(gradJ(theta, xArr1, yArr1, m) )
