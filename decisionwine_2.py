#
#import sys
#sys.modules[__name__].__dict__.clear()

from sklearn import tree # imports the decision tree thing
from sklearn.datasets import load_wine
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd

wine = load_wine()

# remember that you can't print out wine shape, but you can
# print out the data and its dimensionality and data
print(wine.data.shape)

# first two dimesions of wine data
x = wine.data[:,:]

print(x.shape[0])
ntrain = int(np.floor(x.shape[0]*.8)) # cast to an int or else it will float
ntest = x.shape[0] - ntrain


# data points, what it prints is the type number
#print(wine.target)
y = wine.target

# normalize the data?????
print(x.size)
for i in range(0, (x.shape[1] - 1)):
    x[:,i] = (x[:,i] - x[:,i].min())/(x[:,i].max() - x[:,i].min())
    
reps = 50
dcs = tree.DecisionTreeClassifier(criterion="gini")

avgacc = 0

for i in range(0,reps):
    ran = np.random.permutation(np.linspace(0, x.shape[0] - 1, x.shape[0], dtype="int"))
    # put the training data in an array
    trainx = np.array([x[i] for i in ran[:ntrain]])
    trainy = np.array([y[i] for i in ran[:ntrain]])
    dcs.fit(trainx, trainy)
       
    testx = [x[i] for i in ran[ntrain:]]
    testy = [y[i] for i in ran[ntrain:]]
    # compare the real answers to the result
    testz = dcs.predict(testx)
    corr = np.sum([testy[i] == testz[i] for i in range(0,ntest)])/ntest
    print(corr*100)
    avgacc += corr*100
    
avgacc /= reps
print(avgacc)






