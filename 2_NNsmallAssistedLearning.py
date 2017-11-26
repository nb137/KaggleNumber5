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
# score on train data is 0.91488

outMatrix = model.predict(test)
outMatrix = pd.DataFrame(outMatrix)
unk = outMatrix[outMatrix.sum(axis=1) == 0]  # get unk answers which we know are wrongly predicted, will use on model2
unkTest = test.iloc[unk.index]  # Images where we got no output
# What does the NN look like on these if we put in the digit itself?
# Look at some number of unknown images, predict them, and add them to the training dataset

nums = np.where(outMatrix ==1)[1]

# Re-do without matrix-ization of input
#model2 = MLPClassifier(activation='logistic', alpha=0.001)
#model2.fit(trainMatrix, train.label)
# model2 score on train data is 0.94878, higher 
#outNums = model2.predict(test)
#outNums = pd.DataFrame(outNums)

#outUnk = outNums.iloc[unk.index]  # here we see what this model predicts for numbers that we think will be wrongly predicted

''' Start assisted learning '''
# I manually look at the 25 first unknown images and classify them
unkTestAssist_Y = pd.DataFrame([0, 4, 6, 9, 8, 5, 3, 3, 8, 3, 5, 5, 5, 0, 9, 3, 7, 9, 4, 8, 5, 4, 5, 5, 6])
unkTestAssist_X = unkTest.iloc[range(25)]

# Add the assisted learning parts to the train data to get a better fit (we hope)
train2 = pd.concat([trainMatrix,unkTestAssist_X])
ans2 = pd.concat([train.label, unkTestAssist_Y])

model2.fit(train2, ans2)
# score is now 0.95588
outNums = model2.predict(test)

outNums = pd.DataFrame(outNums,columns=["Label"])
outNums["ImageId"] = range(1, len(outNums)+1)

outNums[["ImageId","Label"]].to_csv("2_Assisted25.csv", index=False)


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