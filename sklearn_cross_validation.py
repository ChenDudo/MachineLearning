# !/usr/bin/env python3
# -*- coding:UTF-8 -*-
# author: ChenDu
# time: 2018/11/10 0010
# 
# 
# 用sklearn的交叉验证 训练数据集


import pandas as pd
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier

x = pd.read_csv("iris.csv")
y = x.pop("species")
x_train, x_test, y_train, y_test = model_selection.train_test_split(x.values, y.values, test_size=0.1)

scores = model_selection.cross_val_score(KNeighborsClassifier(3), x, y, cv=5)
mean_score = scores.mean()
print("mean score: "+str(mean_score))
