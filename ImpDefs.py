def JCost(thetaArr, xArr, yArr):
	hypo = hypothesis(thetaArr, xArr)
	J = sum(  (1.0/len(yArr))*(  -1*yArr*np.log(hypo) - (1-yArr)*np.log(1 - hypo)  )  )
	return J

def JECost(thetaArr, xArr, y):
	J = 0
	for j in range(len(y)):
		hypo = hypothesis(thetaArr, xArr)
		J = J + (1.0/len(y))*(  -1*y[j]*log(hypo[j]) - (1-y[j])*log(1 - hypo[j])  )
	return J



def hypothesis(thetaArr, xArr):
	oldhypo = np.matmul(thetaArr, np.transpose(xArr) )
	newhypo = 1/(1+np.exp(-oldhypo))
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

# Calculate the gradJ vector
def gradJ(thetaArr, xArr, yArr):
	hypo = hypothesis(thetaArr, xArr)
	gradj = (1.0/len(yArr))*np.matmul( np.transpose(xArr), hypo - yArr)
	return gradj



# We want to normalize the data. Divide by the standard deviation and subtract the mean. Done in 2 loops so the mean and stdev don't change in the loop. We record these so we can undo the effects later
def NormData(x, xnew):
	xTemp = [0 for i in range(len(x))]
	xStd  = np.std(x)
	xMean = np.mean(x)

	for i in range(len(x)):
		xTemp[i] = (x[i] - xMean )/ xStd
	for i in range(len(x)):
		xnew[i] = xTemp[i]
	return [xMean, xStd]


# Take out one column (column i) from a matrix
def column(matrix, i):
    return [row[i] for row in matrix]











