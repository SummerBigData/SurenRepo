# Written by: 	Suren Gourapura
# Written on: 	May 31, 2018
# Purpose: 	To solve exercise 4 on Multi-class Classification and Neural Networks in Coursera
# Goal:		Create a confusion matrix for the RevProp probabilities

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Assign the data to arrays


imp0 = np.array([0.73314908, 0.00805775, 0.00215607, 0.00091405, 0.00404813, 0.01186563, 0.00141346, 0.00595548, 0.000207, 0.00198292])
imp1 = np.array([ 0.00032019,  0.96241484,  0.00044785, 0.00007297,  0.0022549,   0.01940793, 0.0005833,  0.00242801,  0.00001354,  0.00094198])
imp2 = np.array([ 0.00018648,  0.0011387,   0.99473014,  0.0001149,   0.00128525,  0.00419275, 0.00058668,  0.00129992,  0.00001113,  0.00020219])
imp3 = np.array([ 0.00026444,  0.00592267,  0.0008752,   0.93110739,  0.00405756,  0.00930494, 0.00114766,  0.00605033,  0.00005896,  0.00036863])
imp4 = np.array([ 0.00004302,  0.00125485,  0.001161,    0.00014906,  0.98883836,  0.00384799,  0.00037307,  0.00224442,  0.00001053,  0.00215142])
imp5 = np.array([ 0.00019891,  0.00343341,  0.00035435,  0.00021848,  0.00153284,  0.99563546,  0.00036901,  0.00220887, 0.0000219,   0.00032046])
imp6 = np.array([ 0.00005659,  0.00190516,  0.00111758,  0.00009635,  0.00367413,  0.00859801, 0.82109434,  0.00240647 , 0.00000324,  0.00033466])
imp7 = np.array([ 0.00014128,  0.00261487,  0.00059594,  0.00016392,  0.00163181,  0.00379474, 0.0004515 ,  0.99463583, 0.0000252 ,  0.00029702])
imp8 = np.array([ 0.00010818,  0.00266544,  0.00075211,  0.00009889,  0.00442395,  0.04376797, 0.00355584,  0.0013806  , 0.91797483,  0.00030144])
imp9 = np.array([ 0.00004301,  0.00147344,  0.00040959, 0.00007091 , 0.0037166 ,  0.01413552  ,0.00078946,  0.00218029  ,0.00001246 , 0.97585318])

# Stack the images together vertically
confuMat = np.concatenate(([imp0],[imp1],[imp2],[imp3],[imp4],[imp5],[imp6],[imp7],[imp8],[imp9]), axis = 0)


# Place the highest values together horizontally
barPlot = np.zeros((10))

for i in range(10):
	barPlot[i] = np.amax(confuMat[i])
print barPlot

# DISPLAY PICTURES
img = plt.imshow(confuMat, cmap="coolwarm", interpolation='none') 
plt.colorbar()
plt.savefig('results/ConfusionMatrix.png',transparent=True, format='png')
plt.show()

barplt = plt.bar([0,1,2,3,4,5,6,7,8,9], barPlot,facecolor='red', align='center')
plt.savefig('results/MaxRevBarPlot.png',transparent=False, format='png')
plt.show()




