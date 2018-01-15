from sklearn import tree # imports the decision tree thing
import imageio
import matplotlib.pyplot as plt
import numpy as np


pigeon = np.zeros((288300, 30))
macaw = np.zeros((288300, 30))
for i in range(0, 10):
    # path get - image read!
    im_ma = imageio.imread('C:/Users/Isaac Pang/Downloads/' +
                       'BirdPhotos_ForHPCMentor/Macaw_0' + str(i) + '.jpg')
    im_ma = im_ma[int(im_ma.shape[0]/2 -155):int(im_ma.shape[0]/2) + 155, 
                  int(im_ma.shape[1]/2 -155):int(im_ma.shape[1]/2) + 155, :]
    # stacks up all values into one array; automatically ravels
#    print(len(im_ma.ravel()))
    macaw[:, i] = im_ma.ravel()
    im_pi = imageio.imread('C:/Users/Isaac Pang/Downloads/' +
                           'BirdPhotos_ForHPCMentor/Pigeon_0'+ str(i) + '.jpg')
    im_pi = im_pi[int(im_pi.shape[0]/2) -155:int(im_pi.shape[0]/2) + 155, 
                  int(im_pi.shape[1]/2) -155:int(im_pi.shape[1]/2) + 155, :]    
    pigeon[:, i] = im_pi.ravel()


for i in range(10, 30):
    im_ma = imageio.imread('C:/Users/Isaac Pang/Downloads/' +
                       'BirdPhotos_ForHPCMentor/Macaw_' + str(i) + '.jpg')
    im_ma = im_ma[int(im_ma.shape[0]/2 -155):int(im_ma.shape[0]/2) + 155, 
                  int(im_ma.shape[1]/2 -155):int(im_ma.shape[1]/2) + 155, :]
    macaw[:, i] = im_ma.ravel()
    im_pi = imageio.imread('C:/Users/Isaac Pang/Downloads/' +
                           'BirdPhotos_ForHPCMentor/Pigeon_'+ str(i) + '.jpg')
    im_pi = im_pi[int(im_pi.shape[0]/2) -155:int(im_pi.shape[0]/2) + 155, 
                  int(im_pi.shape[1]/2) -155:int(im_pi.shape[1]/2) + 155, :]
    pigeon[:, i] = im_pi.ravel()


    
print(pigeon.shape)
print(macaw.shape)

bird_data = np.concatenate((macaw, pigeon), axis=1)
print(bird_data.shape)


print(bird_data.shape[0])
ntrain = int(np.floor(bird_data.shape[1]*.8)) # cast to an int or else it will float
ntest = bird_data.shape[1] - ntrain


# data points, what it prints is the type number
bird_target = np.zeros((60,1))
bird_target[30:60, 0] = 1


# normalize the data?????
print(bird_data.size)
#for i in range(0, (bird_data.shape[1] - 1)):
#    bird_data[:,i] = (bird_data[:,i] - bird_data[:,i].min())/(bird_data[:,i].max() 
#                    - bird_data[:,i].min())
    
reps = 50
dcs = tree.DecisionTreeClassifier(min_samples_leaf=4, criterion="gini")

avgacc = 0

for i in range(0,reps):
    perm = np.random.permutation(np.linspace(0, bird_data.shape[1] - 1,
                                            bird_data.shape[1], dtype="int"))
    # put the training data in an array
    trainx = np.array([bird_data[:,i] for i in perm[:ntrain]])
    trainy = np.array([bird_target[i] for i in perm[:ntrain]])
    dcs.fit(trainx, trainy)
       
    testx = [bird_data[:,i] for i in perm[:ntest]]
    testy = [bird_target[i] for i in perm[:ntest]]
    testy = np.concatenate(testy)    
    # compare the real answers to the result
    testz = dcs.predict(testx)
    corr = np.sum([testy[i] == testz[i] for i in range(0,ntest)])/ntest
    print(corr*100)
    avgacc += corr*100
    
avgacc /= reps
print(avgacc)




#plt.imshow(a)
#a_0 = a[:,:,1]
#plt.imshow(a_0)

#plt.imshow(c)
#plt.imshow(c[57:,:,:])
