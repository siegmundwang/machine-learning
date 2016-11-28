#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 17:13:50 2016
titanic 
@author: sig
"""
# %%
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
# %%
tr = pd.read_csv("tt_train.csv")
ts = pd.read_csv("tt_test.csv")
tr.info()
ts.info()
# %%
selected_ft = ['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch','Fare']
X_tr = tr[selected_ft]
X_ts = ts[selected_ft]
y_tr = tr['Survived']
# %%
X_tr['Embarked'].value_counts()
X_ts['Embarked'].value_counts()
X_tr['Embarked'].fillna('S', inplace = True)
X_ts['Embarked'].fillna('S', inplace = True)
X_tr['Age'].fillna(X_tr['Age'].mean(), inplace = True)
X_ts['Age'].fillna(X_ts['Age'].mean(), inplace = True)
X_ts['Fare'].fillna(X_ts['Fare'].mean(), inplace = True)
# %%
print(X_tr.info())
print(X_ts.info())
# %%
dict_vec = DictVectorizer(sparse = False)
X_tr = dict_vec.fit_transform(X_tr.to_dict(orient = 'record'))
dict_vec.feature_names_
X_ts = dict_vec.transform(X_ts.to_dict(orient = 'record'))
# %%
rfc = RandomForestClassifier()
xgbc = XGBClassifier()
cross_val_score(rfc, X_tr, y_tr, cv = 5).mean()
cross_val_score(xgbc, X_tr, y_tr, cv = 5).mean()
# %%
rfc.fit(X_tr, y_tr)
rfc_y_prd = rfc.predict(X_ts)
rfc_submission = pd.DataFrame({'PassengerId': ts['PassengerId'], \
                               'Survived': rfc_y_prd})
rfc_submission.to_csv('rfc_submission.csv', index = False)
# %%
xgbc.fit(X_tr, y_tr)
xgbc_y_prd = xgbc.predict(X_ts)
xgbc_submission = pd.DataFrame({'PassengerId': ts['PassengerId'], \
                               'Survived': xgbc_y_prd})
xgbc_submission.to_csv('xgbc_submission.csv', index = False)
# %%
params = {'max_depth': range(2, 7), 'n_estimators': range(100, 1100, 200), \
          'learning_rate': [0.05, 0.1, 0.25, 0.5, 1.0]}
xgbc_best = XGBClassifier()
gs = GridSearchCV(xgbc_best, params, n_jobs = 1, cv = 5, verbose = 1)
gs.fit(X_tr, y_tr)
# %%
print(gs.best_score_)
print(gs.best_params_)
xgbc_best_y_prd = gs.predict(X_ts)
xgbc_best_submission = pd.DataFrame({'PassengerId': ts['PassengerId'], \
                                     'Survived': xgbc_best_y_prd})
xgbc_best_submission.to_csv('gbc_best_submission.csv', index=False)
