import numpy as np

W = np.array([[i for i in range(4)] for j in range(5)])
b = np.array([[j for i in range(1)] for j in range(5)])

W1 = np.concatenate(( np.ravel(W), np.ravel(b)))
print W, b, W1

WAll = np.hstack((b, W))

print WAll

print WAll[:, :1]
print WAll[:, 1:]
