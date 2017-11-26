#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 21:16:54 2017

@author: nb137
"""

import pandas as pd
import numpy as np


test = pd.read_csv("test.csv")

train = pd.read_csv("train.csv")

answerMatrix = pd.get_dummies(train.label)  # 42000 x 10
trainMatrix = train.drop("label", axis=1)   # 42000 x 784

from sklearn.neural_network import MLPClassifier
model = MLPClassifier(activation='logistic', alpha=0.001)

model.fit(trainMatrix, answerMatrix)

outMatrix = model.predict(test)
outMatrix = pd.DataFrame(outMatrix)

unk = outMatrix[outMatrix.sum(axis=1) == 0]
unkTest = test.iloc[unk.index]  # Images where we got no output
# What does the NN look like on these if we put in the digit itself?

nums = np.where(outMatrix ==1)[1]

# Re-do without matrix-ization of input
model2 = MLPClassifier(activation='logistic', alpha=0.001)
model2.fit(trainMatrix, train.label)
outNums = model2.predict(test)
outNums = pd.DataFrame(outNums)

outUnk = outNums.iloc[unk.index]

outNums = pd.DataFrame(outNums,columns=["Label"])
outNums["ImageId"] = range(1, len(outNums)+1)

outNums[["ImageId","Label"]].to_csv("1_FirstTry.csv", index=False)


'''
import tensorflow as tf
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))    # Input is 784, hidden layer is 10
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)
y_ = tf.placeholder(tf.float32, [None, 10]) # Placeholder for answers

cross_entropy = tf.reduce_mean( -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(200):
    sess.run(train_step, feed_dict = {x:trainMatrix, y_:answerMatrix})
    
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: trainMatrix, y_: answerMatrix}))
'''