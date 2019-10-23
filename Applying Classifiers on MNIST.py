#Running the entire code at once is extremely time consuming.
#Also, I use a training dataset of 12000 entries and testing dataset of 2000 entries
#leading to a lower accuracy than usual.

import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

mnist_train=pd.read_csv("E:\mnist_train.csv").as_matrix()
mnist_test=pd.read_csv("E:\mnist_test.csv").as_matrix()

clfdtc=DecisionTreeClassifier()
clfknn=KNeighborsClassifier()
clfsvm=SVC(kernel='linear')

#training dataset
xtrain=mnist_train[0:12000, 1:]
train_label=mnist_train[0:12000, 0]

clfdtc.fit(xtrain, train_label)
clfknn.fit(xtrain, train_label)
clfsvm.fit(xtrain, train_label)

#testing dataset
xtest=mnist_test[0:2000, 1:]
actual_label=mnist_test[0:2000, 0]

#Sample accuracy test
'''d=xtest[47]
d.shape=(28, 28)
pt.imshow(255-d, cmap='gray')
print(clfdtc.predict( [xtest[47]]) )
print(clfknn.predict( [xtest[47]]) )
pt.show()''' 

#Using Data Tree Classifier
preddtc=clfdtc.predict(xtest)
count=0
for i in range(0, 1000):
    count+=1 if preddtc[i]==actual_label[i] else 0
print("Accuracy as per Data Tree Classifier= ", (count/1000)*100)
#Data Tree Classifier accuracy comes out to be 80.3%

#Using K Nearest Neighbor Classifier
predknn=clfknn.predict(xtest)
count=0
for i in range(0, 1000):
    count+=1 if predknn[i]==actual_label[i] else 0
print("Accuracy as per K Nearest Neighbor= ", (count/1000)*100)
#K Nearest Neighbor accuracy comes out to be 91.6%

#Using SVM Classifier
predsvm=clfsvm.predict(xtest)
count=0
for i in range(0, 1000):
    count+=1 if predsvm[i]==actual_label[i] else 0
print("Accuracy as per SVM= ", (count/1000)*100)
#SVM accuracy comes out to be 90.3%