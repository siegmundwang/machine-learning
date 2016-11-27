#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 20:04:19 2016
decision tree
@author: sig
"""
# %%
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
# %%
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
titanic.head()
titanic.info()
# %%
X = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']
X.info()
X['age'].fillna(X['age'].mean(), inplace = True)
# %%
X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size = 0.25, random_state = 33)
vec = DictVectorizer(sparse = False)
X_tr = vec.fit_transform(X_tr.to_dict(orient = 'record'))
print(vec.feature_names_)
X_tst = vec.transform(X_tst.to_dict(orient = 'record'))
# %%
dtc = DecisionTreeClassifier()
dtc.fit(X_tr, y_tr)
y_prd = dtc.predict(X_tst)
print(dtc.score(X_tst, y_tst))
print(classification_report(y_tst, y_prd, target_names = ['died', 'survived']))
