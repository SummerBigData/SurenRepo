import numpy as np

d3 = np.arange(36).reshape(3,4,3)

print d3
print ' '

raveld3 = np.ravel(d3)

print raveld3.reshape(3,4,3)
