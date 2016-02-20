# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 14:34:02 2015

@author: Jessica
"""
import time
import pandas as pd
#import numpy as np
import os
from sklearn.cross_validation import train_test_split
from sklearn import decomposition
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.lda import LDA
from sklearn.metrics import accuracy_score

#import matplotlib.pyplot as plt

os.chdir('D:\BIA\BIA678 Big Data Seminar\HOMEWORK\FINAL PROJECT')
os.getcwd()

df=pd.read_csv('train.csv')
x=df.ix[:,1:785]
y=df['label']

#split the data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

t0=time.time()

#following are 4 types of dimension reduction
#principal component analysis
"""
tr_func=decomposition.PCA(n_components=0.8,whiten=True)
x_train=tr_func.fit_transform(x_train)


#feature selection
tr_func=SelectKBest(chi2,k=300)
x_train=tr_func.fit_transform(x_train,y_train)
"""
#Truncated SVD
tr_func=decomposition.TruncatedSVD(n_components=43, random_state=42)
x_train=tr_func.fit_transform(x_train,y_train)
"""
#linear discriminant analysis
tr_func=LDA(n_components=15)
x_train=tr_func.fit_transform(x_train,y_train)
"""
#transform the test data
x_test=tr_func.transform(x_test)

t1=time.time()
#modeling
clf=RandomForestClassifier(n_estimators=20,max_features=15,criterion='entropy') 
model=clf.fit(x_train,y_train)
pred=model.predict(x_test)  
accuracy=accuracy_score(y_test,pred)

t2=time.time()  
#ccompute the time spent in each step
print accuracy, t1-t0,t2-t1,t2-t0


