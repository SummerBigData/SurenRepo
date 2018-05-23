import numpy as np

x = np.asarray([[ 1,2,3, 2], [4,5,6, 2] , [7,10,9, 5.5]])
print x
x[2] = np.asarray([1,2,3,4])
print x[2]
print x[2].argmax()
