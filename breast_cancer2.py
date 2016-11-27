#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 01:36:15 2016
more detailed cancer, logistic regression and SGDClassifier
@author: sig
"""
# %% 
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
# %% read data
data = pd.read_csv('cancer-data.csv', index_col=0).iloc[:,:31]
data = data.replace(to_replace='?', value=np.nan)
data = data.dropna(how = 'any')
data.shape
# %% split train and test data
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,1:], \
                                                    data.iloc[:,0], test_size=0.25, random_state=33)
y_train.value_counts()
y_test.value_counts()
# %% standardize data
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
# %% train model
lr = LogisticRegression()
sgdc = SGDClassifier()

lr.fit(X_train, y_train)
lr_y_predict = lr.predict(X_test)

sgdc.fit(X_train, y_train)
sgdc_y_predict = sgdc.predict(X_test)
# %% model report
print('Accuracy of LR Classifier:', lr.score(X_test, y_test))
print(classification_report(y_test, lr_y_predict, target_names=['Benign', 'Malignant']))
print('Accuracy of SGD Classifier', sgdc.score(X_test, y_test))
print(classification_report(y_test, sgdc_y_predict, target_names=['Benign', 'Malignant']))
