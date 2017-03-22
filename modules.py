# -*- Coding: utf-8 -*-

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import datasets

def plot_decision_regions(X,y,classifier,test_idx=None,resolution=0.02):
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:,0].min() -1, X[:,0].max() + 1
    x2_min, x2_max = X[:,1].min() -1, X[:,1].max() + 1
    
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2,Z,alpha=0.1, cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y ==cl, 0],y=X[y==cl, 1],alpha = 0.8, c=cmap(idx),marker =markers[idx],label=cl)
    if test_idx:
        X_test, y_test = X[test_idx,:], y[test_idx]
        plt.scatter(X_test[:,0],X_test[:,1],c='', edgecolors="black", alpha=0.7, linewidths=1, marker='o',s=55,label='test set')

def get_iris_data():
	iris = datasets.load_iris()
	X = iris.data[:,[2,3]]
	y = iris.target
	X_train, X_test, y_train, y_test = train_test_split( X, y,  test_size = 0.3, random_state=0)
	X_combined = np.vstack((X_train, X_test))
	y_combined = np.hstack((y_train, y_test))
	return X_train, X_test, X_combined, y_train, y_test, y_combined