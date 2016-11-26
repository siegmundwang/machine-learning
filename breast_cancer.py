#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 00:46:27 2016
my first sklearn script for cancer diagnosis
@author: sig
"""
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression 

df_train = pd.read_csv("breast-cancer-train.csv")
df_test = pd.read_csv("breast-cancer-test.csv")
df_test_negative = df_test[df_test["Type"] == 0][["Clump Thickness", "Cell Size"]]
df_test_positive = df_test[df_test["Type"] == 1][["Clump Thickness", "Cell Size"]]

# %%
intercept = np.random.random([1])
coef = np.random.random([2])
lx = np.arange(12)
ly = (-intercept - lx * coef[0])/coef[1]

plt.plot(lx, ly, c = "yellow")
plt.scatter(df_test_negative["Clump Thickness"], df_test_negative["Cell Size"], 
            marker = "x", s = 200, c = "red")
plt.scatter(df_test_positive["Clump Thickness"], df_test_positive["Cell Size"], 
            marker = "o", s = 150, c = "black")
plt.xlabel("Clump Thickness")
plt.ylabel("Cell Size")
plt.show()
# %%
lr = LogisticRegression()
lr.fit(df_train[['Clump Thickness', 'Cell Size']][:10], df_train["Type"][:10])
print("Test accuracy:", lr.score(df_test[['Clump Thickness', 'Cell Size']],
                                 df_test["Type"]))

intercept = lr.intercept_
coef = lr.coef_[0, :] # flatten
ly = (-intercept - lx * coef[0])/coef[1] # mapping from 3D to 2D
plt.plot(lx, ly, c = "green")
plt.scatter(df_test_negative["Clump Thickness"], df_test_negative["Cell Size"], 
            marker = "x", s = 200, c = "red")
plt.scatter(df_test_positive["Clump Thickness"], df_test_positive["Cell Size"], 
            marker = "o", s = 150, c = "black")
plt.xlabel("Clump Thickness")
plt.ylabel("Cell Size")
plt.show()
# %% use all sample
lr = LogisticRegression()
lr.fit(df_train[['Clump Thickness', 'Cell Size']], df_train["Type"])
print("Test accuracy:", lr.score(df_test[['Clump Thickness', 'Cell Size']],
                                 df_test["Type"]))
intercept = lr.intercept_
coef = lr.coef_[0, :] # flatten
ly = (-intercept - lx * coef[0])/coef[1] # mapping from 3D to 2D
plt.plot(lx, ly, c = "green")
plt.scatter(df_test_negative["Clump Thickness"], df_test_negative["Cell Size"], 
            marker = "x", s = 200, c = "red")
plt.scatter(df_test_positive["Clump Thickness"], df_test_positive["Cell Size"], 
            marker = "o", s = 150, c = "black")
plt.xlabel("Clump Thickness")
plt.ylabel("Cell Size")
plt.show()
