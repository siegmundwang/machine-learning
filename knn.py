#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 19:51:09 2016
k-NN
@author: sig
"""
# %%
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
# %%
iris = load_iris()
iris.data.shape
print(iris.DESCR)
X_tr, X_tst, y_tr, y_tst = train_test_split(iris.data, iris.target, \
                                            test_size = 0.25, \
                                            random_state = 33)
# %%
ss = StandardScaler()
X_tr = ss.fit_transform(X_tr)
X_tst = ss.transform(X_tst)
knc = KNeighborsClassifier()
knc.fit(X_tr, y_tr)
y_prd = knc.predict(X_tst)
print('The accuracy of K-Nearest Neighbor Classifier is', knc.score(X_tst, y_tst))
print(classification_report(y_tst, y_prd, target_names = iris.target_names))
