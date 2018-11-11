# !/usr/bin/env python3
# -*- coding:UTF-8 -*-
# author: ChenDu
# time: 2018/11/10 0010
# 
# 用sklearn的交叉验证 KNN 逻辑蒂斯回归 三种方式 训练数据集 并对比


import pandas as pd
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

x = pd.read_csv("iris.csv")
y = x.pop("species")
x_train, x_test, y_train, y_test = model_selection.train_test_split(x.values, y.values, test_size=0.1)

models = {
    "knn": KNeighborsClassifier(6),
    "gnb": GaussianNB(),
    "lr": LogisticRegression(multi_class="multinomial", solver="lbfgs")
}

for name, model in models.items():
    score = model_selection.cross_val_score(model, x, y, cv=5).mean()
    print(name, score)
