# !/usr/bin/env python3
# -*- coding:UTF-8 -*-
# author: ChenDu
# time: 2018/11/10 0010
#
# 用sklearn的朴素贝叶斯 训练数据集


import pandas as pd
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB

x = pd.read_csv("iris.csv")
y = x.pop("species")
x_train, x_test, y_train, y_test = model_selection.train_test_split(x.values, y.values, test_size=0.1)

gnb = GaussianNB().fit(x_train, y_train)
print(gnb.predict_proba(x_test))
print(gnb.class_prior_)
