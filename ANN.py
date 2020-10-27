#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 08:18:02 2020

@author: kiran
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1]
y = dataset.iloc[:, -1]

Geography = pd.get_dummies(X['Geography'], drop_first= True)
Gender = pd.get_dummies(X['Gender'], drop_first= True)

X = X.drop(['Geography','Gender'], axis = 1)

X = pd.concat([X, Geography, Gender], axis = 1)

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X = sc_X.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.2 , random_state = 0)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import ReLU, LeakyReLU
from keras.layers import Dropout

classifier = Sequential()

classifier.add(Dense(units = 6 , activation = 'relu', kernel_initializer = 'he_uniform', input_dim = 11))
classifier.add(Dropout(0.2))

classifier.add(Dense(units = 6, activation= 'relu', kernel_initializer = 'he_uniform'))
classifier.add(Dropout(0.3))

classifier.add(Dense(units= 1, activation= 'sigmoid', kernel_initializer= 'glorot_uniform'))

classifier.compile(optimizer= 'adamax', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier_history = classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100, validation_split = 0.33, shuffle = True)

print(classifier_history.history.keys())

plt.plot(classifier_history.history['loss'])
plt.plot(classifier_history.history['val_loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc = 'upper right')
plt.show()

plt.plot(classifier_history.history['accuracy'])
plt.plot(classifier_history.history['val_accuracy'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc = 'upper right')
plt.show()

y_pred = classifier.predict(X_test)
y_pred = y_pred > 0.5

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)


from sklearn.metrics import accuracy_score

score = accuracy_score(y_test, y_pred)


print(classifier_history.history.keys())