# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 12:26:37 2017

@author: bdabdoub
"""


import warnings
import numpy as numpy
import pandas as pandas
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Returns a three-way split of the data
def train_validation_test_split(X, y, trainSize, validationSize, testSize,):
    X_train, X_remainder, y_train, y_remainder = train_test_split(X, y, test_size=testSize+validationSize, random_state=0)
    X_validation, X_test, y_validation, y_test = train_test_split(X_remainder, y_remainder, test_size=0.5, random_state=0)
    return (X_train, X_validation, X_test, y_train, y_validation, y_test)


df = pandas.read_csv('titanicdata.csv')

#Drop examples with null values
df = df.dropna(subset =['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'])
#Convert attribute values to numeric type
df.apply(pandas.to_numeric, errors='ignore')

#Read in y values, convert to 0 or 1
y = df.iloc[:, 1].values
y = numpy.where(y == 0, 0, 1)

#read in X values, convert sex to 0 or 1
X = df.iloc[:, [2, 4, 5, 6, 7, 9]].values
X[X == 'male'] = 0
X[X == 'female'] = 1

#Print the possible class labels
print('Class labels:', numpy.unique(y))

#Split the data into train, validation, and test sets
X_train, X_validation, X_test, y_train, y_validation, y_test = train_validation_test_split(X, y, 0.7, 0.15, 0.15)

#standardiize the data, ignoring type conversion warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_validation_std = sc.transform(X_validation)
    X_test_std = sc.transform(X_test)

#Create a LogisticRegression object, fit it to the training data
lr = LogisticRegression(C=1000, random_state=0)
lr.fit(X_train_std, y_train)

#Predict the label of the validation set
y_validation_pred = lr.predict(X_validation_std)

print('Misclassified samples: %d' % (y_validation != y_validation_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_validation, y_validation_pred))

y_test_pred = lr.predict(X_test_std)

print('Misclassified samples: %d' % (y_test != y_test_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_test_pred))






