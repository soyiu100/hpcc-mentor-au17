
from sklearn import neighbors
from sklearn.datasets import load_wine
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

wine = load_wine()
# this prints a coordinate with the number of data points & the dimensionality
# shape is the size of the data
print(wine.data.shape)

# list of all types
print(wine.target_names)
# list of all dimensions
print(wine.feature_names)

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
# looking at columns 2 and 3 of dimensions 
X = wine.data[:, 4:6]
#training 80% of the data
ntrain = int(np.floor(X.shape[0]*0.8))
# this is the test for the rest of the data
ntest = X.shape[0]-ntrain
y = wine.target
X[:, 0] = (X[:, 0]-X[:,0].min())/(X[:,0].max()-X[:,0].min())
X[:, 1] =(X[:,1]-X[:, 1].min())/(X[:, 1].max() -X[:, 1].min())

reps = 20

avgacc_rec = np.array([np.arange(reps), np.arange(reps, dtype='float')])

for i in range(0,reps):
    perm = np.random.permutation(np.linspace(0,X.shape[0]-1,X.shape[0],dtype='int'))
    knn = neighbors.KNeighborsClassifier(n_neighbors=10)
    trainx = np.array([X[i] for i in perm[:ntrain]])
    trainy = np.array([y[i] for i in perm[:ntrain]])
    knn.fit(trainx,trainy)
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
    np.linspace(y_min, y_max, 100))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    # Plot also the training points
    plt.scatter(trainx[:, 0], trainx[:, 1], c=trainy, cmap=cmap_bold)
    plt.axis('tight')
    
    testx = [X[i] for i in perm[ntrain:]]
    testy = [y[i] for i in perm[ntrain:]]
    testz = knn.predict(testx)
    corr = np.sum([testy[i] == testz[i] for i in range(0,ntest)])/ntest
    avgacc_rec[1,i] = corr*100;
    print(corr*100)

np.average(avgacc_rec[1,:])
