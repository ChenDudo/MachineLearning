# 用sklearn的KNN 训练数据集
import pandas as pd
from sklearn import model_selection
# !/usr/bin/env python3
# -*- coding:UTF-8 -*-
# author: ChenDu
# time: 2018/11/10 0010


from sklearn.neighbors import KNeighborsClassifier

x = pd.read_csv("iris.csv")
y = x.pop("species")
x_train, x_test, y_train, y_test = model_selection.train_test_split(x.values, y.values, test_size=0.1)

knn = KNeighborsClassifier(3).fit(x_train, y_train)
for y_pred, y_true in zip(knn.predict(x_test), y_test):
    print(y_pred, y_true)
print("Knn score:" + str(knn.score(x_test, y_test)))
