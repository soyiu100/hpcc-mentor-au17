
from sklearn import neighbors
from sklearn.datasets import load_wine
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from pandas.plotting import scatter_matrix

# wine data
wine = load_wine()

# these are all of the data points x dimensions
X = wine.data
# number of amount of data being trained
ntrain = int(np.floor(X.shape[0]*0.8))
# number of data being tested
ntest  = X.shape[0]-ntrain

# classes (as numbers) for the wine data
y = wine.target

# normalizing the wine data ((X - X.min())/(X.max() - X.min()))
for i in range(0, wine.data.shape[1]):    
    X[:, i] = (X[:, i]-X[:,i].min())/(X[:,i].max()-X[:,i].min())


# map colorings
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# put all of the data into a dataframe
data = pd.DataFrame(X, index=wine.target)

#now plot using pandas 
#scatter_matrix(data, alpha=0.2, figsize=(10, 10), diagonal='kde')
nxt = 50
knn = neighbors.KNeighborsClassifier(n_neighbors=10)
avgacc = 0

for i in range(0, nxt):
    perm = np.random.permutation(ntrain)
    trainx = np.array([X[i] for i in perm[:ntrain]])
    trainy = np.array([y[i] for i in perm[:ntrain]])
    knn.fit(trainx, trainy)
    
    testx = np.array([X[i] for i in perm[:ntest]])
    testy = np.array([y[i] for i in perm[:ntest]])
    # compare the real answers to the result
    testz = knn.predict(testx)
    corr = np.sum([testy[i] == testz[i] for i in range(0,ntest)])/ntest
    print(corr*100)
    avgacc += corr*100
    
avgacc /= nxt
print(avgacc)    
