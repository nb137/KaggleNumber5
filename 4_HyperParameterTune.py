#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 10:55:08 2017

@author: nb137

In file 3_ we found that KNN made the best fit, but took a long time for predictions
CNN was a little worse but is faster overall

In this file we'll tune the parameters on the training dataset
"""


import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier

train = pd.read_csv("train.csv")

# Try half of the data points to speed up param search
fitData = train.sample(frac=0.5)
X_fit = fitData.drop('label', axis=1)
y_fit = fitData.label

nnModel = MLPClassifier()

params = {
        "activation": ["relu","logistic"],
        "hidden_layer_sizes": [(10),(100)],
        "alpha": [0.0001, 0.01, 0.1],
        "learning_rate": ["constant","adaptive"]}

searchModel = RandomizedSearchCV(nnModel, params, verbose=5)
searchModel.fit(X_fit, y_fit)
# When running on 50% data, the top results were:
# 100 hidden layer, logistic activation
# Unk effect on alpha and learning rate, let's grid search those on full dataset

from sklearn.model_selection import GridSearchCV
nnModel2 = MLPClassifier(hidden_layer_sizes=(100), activation="logistic")
params2 = {
        "alpha": [0.0001, 0.01, 0.1],
        "learning_rate": ["constant","adaptive"]}
searchModel2 = GridSearchCV(nnModel2, params2, verbose=5)
searchModel2.fit(train.drop('label',axis=1), train.label)
# The grid search results aren't drastically different, but the top two scores use alpha = 0.0001
# Adaptive vs. constant has a difference of .003 in favor of constant

# It seems to be using samples of the train data, but to check, lets run two models with adapt. vs. const. using our own train/test
X = train.drop('label', axis=1)
y = train.label
X_test, X_train, y_test, y_train = train_test_split(X,y)

# Adaptive learning
adNN = MLPClassifier(hidden_layer_sizes=(100), activation="logistic", alpha=0.0001, learning_rate="adaptive")
adNN.fit(X_train, y_train)
y_predAd = adNN.predict(X_test)
adScore = accuracy_score(y_test, y_predAd)
# score is 0.923

# Constant learning
clNN = MLPClassifier(hidden_layer_sizes=(100), activation="logistic", alpha=0.0001, learning_rate="constant")
clNN.fit(X_train, y_train)
y_predcl = clNN.predict(X_test)
clScore = accuracy_score(y_test, y_predcl)
# score is 0.9289
# top score from grid search was 0.9401
# All of these are also the default values except the activation function

# What if we're over fitting? Is it any better if we drop hidden layers to ten?
less = MLPClassifier(hidden_layer_sizes=(10), activation="logistic", alpha=0.0001, learning_rate="constant")
less.fit(X_train, y_train)
y_pred = less.predict(X_test)
lessScore = accuracy_score(y_test, y_pred)
# Score is 0.83. noticeably less...

# What if we add another layer
more = MLPClassifier(hidden_layer_sizes=(100,10), activation="logistic", alpha=0.0001, learning_rate="constant")
more.fit(X_train, y_train)
y_pred = more.predict(X_test)
moreScore = accuracy_score(y_test, y_pred)
# Score is 0.921, no evidence that the extra computation time is worth it 