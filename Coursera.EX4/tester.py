
import numpy as np

def Linearize(a, b):
	return np.concatenate((np.ravel(a), np.ravel(b)))

def Unlinearize(vec, a1, a2, b1, b2):
	a = vec[0:a1*a2]
	b = vec[a1*a2:len(vec)]
	a = np.reshape(a, (a1, a2))
	b = np.reshape(b, (b1, b2))
	return a, b


yarr = np.asarray([[i+3*j for i in range(4)] for j in range(7)])
#print yarr

hypo = np.asarray([[i+8*j for i in range(8)] for j in range(4)])
#print hypo
arr = np.asarray([1,3,5,6,2,4,6])
vec = Linearize(yarr, hypo)

a, b = Unlinearize(vec, 7, 4, 4, 8)
Vec = np.array([0 for i in range(5)])
print Vec
print np.insert(Vec, 1, 5)
