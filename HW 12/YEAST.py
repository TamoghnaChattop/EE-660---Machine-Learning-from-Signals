# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 15:44:27 2018

@author: tchat
"""

import csv
import numpy as np 
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.utils import resample
from sklearn.utils import shuffle
import math
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

x_train = pd.read_csv(r'D:\EE 660\HW 12\x_train.csv')
y_train = pd.read_csv(r'D:\EE 660\HW 12\y_train.csv',header=None)
y_train = np.ravel(y_train)
x_test = pd.read_csv(r'D:\EE 660\HW 12\x_test.csv')
y_test = pd.read_csv(r'D:\EE 660\HW 12\y_test.csv',header=None)
y_test = np.ravel(y_test)

acc_test = np.zeros((10,30))
acc_train = np.zeros((10,30))

for b in range(1,31):
    for x in range(1,10):

        X_train_bagged, X_test_bagged, y_train_bagged, y_test_bagged = train_test_split(x_train, y_train, train_size=0.333)
        
        clf = RandomForestClassifier(n_estimators=b, max_features=3, bootstrap=True)
        clf.fit(X_train_bagged, y_train_bagged)
        y_pred_train = clf.predict(X_train_bagged)
        y_pred_test = clf.predict(x_test) 
        
        acc_test[x-1][b-1] = accuracy_score(y_pred_test,y_test)
        acc_train[x-1][b-1] = accuracy_score(y_pred_train,y_train_bagged)
    

mean_accur_test = acc_test.mean(0)
mean_accur_train = acc_train.mean(0)
std_accur_test = acc_test.std(0)
std_accur_train = acc_train.std(0)

print(mean_accur_test)
print(mean_accur_train)
print(acc_test.std(0))
print(acc_train.std(0))

#plt.figure(1)
plt.plot(1-mean_accur_train, 'bo',label='mean_train_accuracy')
#plt.legend(loc='best')
#plt.title('mean_train_accuracy vs no of trees')
#plt.xlabel('No. of trees')
#plt.ylabel('Value of accuracy')

#plt.figure(2)
plt.plot(1-mean_accur_test, 'ro',label='mean_test_accuracy')
#plt.legend(loc='best')
#plt.title('mean_test_accuracy vs no of trees')
#plt.xlabel('No. of trees')
#plt.ylabel('Value of accuracy')

#plt.figure(3)
plt.plot(std_accur_test,'go',label='std_test_accuracy')
#plt.legend(loc='best')
#plt.title('std_test_accuracy vs no of trees')
#plt.xlabel('No. of trees')
#plt.ylabel('Value of accuracy')

#plt.figure(4)
plt.plot(std_accur_train,'co',label='std train accuracy')
#plt.legend(loc='best')
#plt.title('std train accuracy vs no of trees')
#plt.xlabel('No. of trees')
#plt.ylabel('Value of accuracy')

plt.show()