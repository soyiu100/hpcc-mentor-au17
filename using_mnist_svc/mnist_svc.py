# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:36:52 2017

@author: Isaac Pang
"""
import tempfile
import os
import pickle
from pathlib import Path
import scipy.io as scio
from sklearn.datasets import fetch_mldata
from sklearn.svm import SVC
import numpy as np

default_run = True;

if default_run: 
    if os.path.isfile(r'/gscratch/stf/soyiu100/mnist_run/mnist-original.mat'):
        path = r'/gscratch/stf/soyiu100/mnist_run/mnist-original.mat'
        numbers = scio.loadmat(path)
    elif not os.path.exists('path_save.p'):
        mnist_data_home = tempfile.mkdtemp()
        numbers = fetch_mldata('MNIST original', data_home=mnist_data_home)
        mnist_data_home = Path(mnist_data_home)
        new_path = mnist_data_home / 'mldata'
        pickle.dump(new_path, open("path_save.p", "wb"))
    else:
        mnist_data_home = pickle.load(open(r'path_save.p', 'rb'))
        files = os.load_files(new_path)
        numbers = scio.loadmat(files[0])
        
    print(numbers['data'].shape)

    train_coeff = 0.8

    X = numbers['data']
    ntrain = int(np.floor(X.shape[1]*train_coeff))
#    ntest  = X.shape[1]-ntrain
    ntest = int(np.floor(ntrain*.2))   
    
    y = numbers['label']
    
    svc = SVC(C=10.0, degree=5, gamma='auto', kernel='poly', probability=True)
    
    avgacc = 0
    
    reps = 50
    for i in range(0, reps):
        perm = np.random.permutation(X.shape[1])
        trainx = np.array([X[:,i] for i in perm[:ntrain]])
        trainy = np.array([y[:,i] for i in perm[:ntrain]]).ravel()
        svc.fit(trainx, trainy)
        
        
        testx = np.array([X[:,i] for i in perm[ntrain:ntrain+ntest]])
        testy = np.array([y[:,i] for i in perm[ntrain:ntrain+ntest]]).ravel()
        # compare the real answers to the result
        testz = svc.predict(testx)
        corr = np.sum([testy[i] == testz[i] for i in range(0,ntest)])/ntest
        print(corr*100)
        avgacc += corr*100
        
    avgacc /= reps
    print(avgacc)
    
else:
    
    mnist_data_home = tempfile.mkdtemp()
    numbers = fetch_mldata('MNIST original', data_home=mnist_data_home)
    
    print(numbers.data.shape)
    
    train_coeff = 0.8
    X = numbers.data
    ntrain = int(np.floor(X.shape[1]*train_coeff))
    ntest  = X.shape[1]-ntrain
    
    
    y = numbers.target
    
    svc = SVC(C=10.0, degree=5, gamma='auto', kernel='rbf', probability=True)
    
    avgacc = 0
    
    reps = 5
    for i in range(0, reps):
        perm = np.random.permutation(ntrain)
        trainx = np.array([X[i] for i in perm[:ntrain]])
        trainy = np.array([y[i] for i in perm[:ntrain]])
        svc.fit(trainx, trainy)
        
        svc.predict(ntest)
        
        testx = [X[i] for i in perm[ntrain:]]
        testy = [y[i] for i in perm[ntrain:]]
        # compare the real answers to the result
        testz = svc.predict(testx)
        corr = np.sum([testy[i] == testz[i] for i in range(0,ntest)])/ntest
        print(corr*100)
        avgacc += corr*100
        
    avgacc /= reps
    print(avgacc)
        






