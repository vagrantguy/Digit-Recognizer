# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 23:16:56 2015

@author: Ryding
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn import metrics
train = open('C:/Users/helen/Downloads/train_3.csv')

n=0

x=[]
y=[]

for line in train:
    n=n+1
    if n<2:
        continue
    if n > 5000:
        break
    line=line.strip().split(',')
    label=line[0]
    point=line[2:783]
#    p=[]
#    for i in point:
#        if int(i) >=90:
#            p.append(3)
#        elif (int(i)>30 and int(i) <90):
#            p.append(2)
#        elif int(i)==0:
#            p.append(0)
#        else:
#            p.append(1)
#        
#    x.append(p)
    x.append(point)
    y.append(label)
    

#MD=svm.SVC()
MD=RandomForestClassifier()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

MD.fit(x_train,y_train)
pred = MD.predict(x_test)

print "classification accuracy on 70/30 split:", metrics.accuracy_score(y_test, pred)
