import scipy.io
import numpy as np
data = scipy.io.loadmat("ex3weights.mat")

for i in data:
	if '__' not in i and 'readme' not in i:
		np.savetxt(("ex3weights.csv"),data[i],delimiter=',')
