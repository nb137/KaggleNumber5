#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 23:23:36 2017

@author: nb137
"""

# SVC, NN, KNN

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score

train = pd.read_csv("train.csv")
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =  train_test_split(train.drop("label", axis=1),train.label)

# Is this too much for a random forest?
from sklearn.ensemble import RandomForestClassifier
# Come back and try BaggingClassifier? https://stackoverflow.com/questions/31681373/making-svm-run-faster-in-python
rfModel = RandomForestClassifier(n_jobs=4)
start = datetime.now()
rfModel.fit(X_train, y_train)
print("rf Computation Time:")
print(datetime.now()-start)
start=datetime.now()
y_predict = rfModel.predict(X_test)
rfScore = accuracy_score(y_test, y_predict)
print("Fit Time:")
print(datetime.now()-start)
# Fit takes 1 second
# Score of 0.9314


from sklearn.neighbors import KNeighborsClassifier
knnModel = KNeighborsClassifier(n_jobs=4)
start = datetime.now()
knnModel.fit(X_train, y_train)
print("KNN Computation Time:")
print(datetime.now()-start)
y_predict = knnModel.predict(X_test)
knnScore = accuracy_score(y_test, y_predict)
print("Computation and Fit Time:")
print(datetime.now()-start)
# Training took 4 seconds
# Testing took 10+ minutes !??!!
# Score is 0.9651
# With 4 parallel jobs, training was 4 sec, testing was 4 min
# Score 0.9651

from sklearn.neural_network import MLPClassifier
NNmodel = MLPClassifier(activation='logistic', alpha=0.001)
start = datetime.now()
NNmodel.fit(X_train, y_train)
print("CNN Computation Time:")
print(datetime.now()-start)
y_predict = NNmodel.predict(X_test)
print("Computation and Prediction Time:")
print(datetime.now()-start)
NNScore = accuracy_score(y_test, y_predict)
# Training took 38 seconds
# Testing took <1s
# Accuracy is 0.9342