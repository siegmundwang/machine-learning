#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 20:04:19 2016
trees compare
@author: sig
"""
# %%
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
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
rfc = RandomForestClassifier()
gbc = GradientBoostingClassifier()

dtc.fit(X_tr, y_tr)
rfc.fit(X_tr, y_tr)
gbc.fit(X_tr, y_tr)

dtc_prd = dtc.predict(X_tst)
rfc_prd = rfc.predict(X_tst)
gbc_prd = gbc.predict(X_tst)

print("Accuracy of dtc is: ", dtc.score(X_tst, y_tst))
print(classification_report(y_tst, dtc_prd, target_names = ['died', 'survived']))
print("Accuracy of rfc is: ", rfc.score(X_tst, y_tst))
print(classification_report(y_tst, rfc_prd, target_names = ['died', 'survived']))
print("Accuracy of gbc is: ", gbc.score(X_tst, y_tst))
print(classification_report(y_tst, gbc_prd, target_names = ['died', 'survived']))
