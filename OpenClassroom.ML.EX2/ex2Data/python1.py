# Import the modules
import numpy as np
import matplotlib.pyplot as plt
import math

# Obtain the x and y data values and convert them from arrays to lists
xarr = np.genfromtxt("ex2x.dat", dtype=float)
yarr = np.genfromtxt("ex2y.dat", dtype=float)
x = xarr.tolist()
y = yarr.tolist()

# Initialize the theta and temptheta lists we will use
theta = [0,0]
temptheta = [0,0]

# Iterate the following for range(__) generations. (in this case, 200)
for g in range(200):
    
    # Fill the temporary theta list for the y-int 
    temptheta[0] = theta[0] - 0.07 * (1.0 / len(x)) * sum([(theta[0] + theta[1] * x[n] - y[n]) * 1 for n in range(len(x))])

    # Fill the temporary theta list for the slope
    temptheta[1] = theta[1] - 0.07 * (1.0 / len(x)) * sum([(theta[0] + theta[1] * x[n] - y[n]) * x[n] for n in range(len(x))])     

    # equate temptheta to theta so that the loop iterate and make progress
    for j in range(2):
        theta[j] = temptheta[j]

# To display the result, we create a list with points from the calculated line equation
y_guess = [0 for j in range(len(x))]

for n in range(len(x)):
    y_guess[n] = theta[0]*1 + theta[1]*x[n]

# For sanity, print out the y-int and slope
print(theta[0])
print(theta[1])

# Plot the result using matplotlib. Data is blue dots, and our line is a red dashed line
plt.plot(x,y, 'o', x, y_guess, 'r--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient Descent fit after 200 generations')
plt.legend(['Data', 'Gradient Descent'])
plt.show()


