# !/usr/bin/env python3
# -*- coding:UTF-8 -*-
# author: ChenDu
# time: 2018/10/21 0021

# from sklearn import datasets
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt
#
# #使用以后的数据集进行线性回归（这里是波士顿房价数据）
# loaded_data=datasets.load_boston()
# data_X=loaded_data.data
# data_y=loaded_data.target
#
# model=LinearRegression()
# model.fit(data_X,data_y)
#
# print(model.predict(data_X[:4,:]))
# print(data_y[:4])
#
# #使用生成线性回归的数据集，最后的数据集结果用散点图表示
# X,y=datasets.make_regression(n_samples=100,n_features=1,n_targets=1,noise=10)   #n_samples表示样本数目，n_features特征的数目  n_tragets  noise噪音
# plt.scatter(X,y)
# plt.show()
# ---------------------
# 作者：yealxxy
# 来源：CSDN
# 原文：https://blog.csdn.net/u014248127/article/details/78885180
# 版权声明：本文为博主原创文章，转载请附上博文链接！

# from sklearn import datasets
#
# # iris = datasets.load_iris()
# iris = datasets.load_iris()
# digits = datasets.load_digits()
#
# # print(digits.data)
# # print(digits.target)
# print(iris.data)


from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#使用以后的数据集进行线性回归（这里是波士顿房价数据）
loaded_data=datasets.load_boston()
data_X=loaded_data.data
data_y=loaded_data.target

model=LinearRegression()
model.fit(data_X,data_y)

print(model.predict(data_X[:4,:]))
print(data_y[:4])

#使用生成线性回归的数据集，最后的数据集结果用散点图表示
X,y=datasets.make_regression(n_samples=100,n_features=1,n_targets=1,noise=10)   #n_samples表示样本数目，n_features特征的数目  n_tragets  noise噪音
plt.scatter(X,y)
plt.show()
