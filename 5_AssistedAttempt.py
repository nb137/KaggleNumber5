#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 12:38:39 2017

@author: nb137
Build on the parameter tuning in 4_, which really didn't do much improvement over the default settings
Also use method in 2_ where we tried assisted learning

Attempt multiple layers of assisted learning, where we find predictiouns without an answer, look at the img, and feed back into the model

Result: after the first feedback of AL, the second model has MORE unknown predictions. We just made it worse. That's not good.
"""

import pandas as pd
import numpy as np
from matplotlib.pyplot import imshow


test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

answerMatrix = pd.get_dummies(train.label)  # 42000 x 10
trainMatrix = train.drop("label", axis=1)   # 42000 x 784

from sklearn.neural_network import MLPClassifier
model = MLPClassifier(activation='logistic')
model.fit(trainMatrix, answerMatrix)
model.score(trainMatrix, answerMatrix)  # score 0.91923

preds1 = model.predict(test)
# Find images where model made no predictions
preds1 = pd.DataFrame(preds1)
unkAnsIndices = preds1[preds1.sum(axis=1) == 0].index  # DF of rows where sum == 0
unkTestImages = test.iloc[unkAnsIndices]
# There are 1364 test images with no answer on this iteration

# Do manual image classification iterating like so:
assistClass = []
imshow(unkTestImages.iloc[0].reshape(28,28))
assistClass.append[0]
# From doing the classification, the first 25 unk images are:
assist_Y = pd.DataFrame([0, 4, 6, 9, 8, 5, 3, 3, 8, 3, 5, 5, 5, 0, 9, 3, 7, 9, 4, 8, 5, 4, 5, 5, 6])
assist_X = unkTestImages.iloc[range(25)]

# Add these to the previous test sets:
train2 = pd.concat([trainMatrix, assist_X], ignore_index=True)
ans2 = pd.concat([train.label,assist_Y], ignore_index=True)
ans2M = pd.get_dummies(ans2[0])
# instead of concat you could set the MLPC model to warm_start=True and add the new answers to go on top of the previous fitr

# Train again with added images
model.fit(train2,ans2M)
model.score(trainMatrix, answerMatrix)  # score 0.90988 getting worse...

preds2 = model.predict(test)
preds2 = pd.DataFrame(preds2)
unkAnsIndices2 = preds2[preds2.sum(axis=1) == 0].index  # DF of rows where sum == 0
unkTestImages2 = test.iloc[unkAnsIndices2]
# Now there are 1541 unknown images, that made things ~10% worse?

# Are we overfitting? Try with a higher alpha and see how many unknowns we get
model2 = MLPClassifier(activation='logistic', alpha = 1)
model2.fit(trainMatrix, answerMatrix)
model2.score(trainMatrix, answerMatrix) # score is 0.8893
predsNew = model2.predict(test)
predsNew = pd.DataFrame(predsNew)
unkAnsIndicesNew = predsNew[predsNew.sum(axis=1) == 0].index  # DF of rows where sum == 0
unkTestImagesNew = test.iloc[unkAnsIndicesNew]
# There are now 2449 images with unknown predictions. A whole lot worse.
