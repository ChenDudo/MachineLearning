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

localfn='./iris.data'

from numpy import genfromtxt, zeros

data = genfromtxt(localfn,delimiter=',',usecols=(0,1,2,3))

target = genfromtxt(localfn,delimiter=',',usecols=(4),dtype=str)

print(data.shape)
print(target.shape)

print(set(target))

from pylab import plot, show
plot(data[target=='setosa',0],data[target=='setosa',2],'bo')
plot(data[target=='versicolor',0],data[target=='versicolor',2],'r+')
plot(data[target=='virginica',0],data[target=='virginica',2],'g*')
show()

setosa_sepal_x=ssx=data[target=='setosa',0]
setosa_sepal_y=ssy=data[target=='setosa',1]
setosa_petal_x=spx=data[target=='setosa',2]
setosa_petal_y=spy=data[target=='setosa',3]

versicolor_sepal_x=vsx=data[target=='versicolor',0]
versicolor_sepal_y=vsy=data[target=='versicolor',1]
versicolor_petal_x=vpx=data[target=='versicolor',2]
versicolor_petal_y=vpy=data[target=='versicolor',3]

virginica_sepal_x=vgsx=data[target=='virginica',0]
virginica_sepal_y=vgsy=data[target=='virginica',1]
virginica_petal_x=vgpx=data[target=='virginica',2]
virginica_petal_y=vgpy=data[target=='virginica',3]

plot(ssx,ssy,'bo',spx,spy,'b+')
plot(vsx,vsy,'ro',vpx,vpy,'r+')
plot(vgsx,vgsy,'go',vgpx,vgpy,'g+')
show()


#figure for 1D（花萼的长度），三类长度及平均值的直方图
#pylab详细用法参考如下
#http://hyry.dip.jp/tech/book/page/scipy/matplotlib_fast_plot.html
from pylab import figure, subplot, hist, xlim, show
xmin = min(data[:,0])
xmax = max(data[:,0])
figure() #可省略，默认会生成一个figure
subplot(411) # distribution of the setosa class (1st, on the top)
hist(data[target=='setosa',0],color='b',alpha=.7)
xlim(xmin,xmax)
#subplot（行,列,plot号）；(4,1,2)合并为412,都小于10可合成
subplot(412) # distribution of the versicolor class (2nd)
hist(data[target=='versicolor',0],color='r',alpha=.7)
xlim(xmin,xmax)
subplot(413) # distribution of the virginica class (3rd)
hist(data[target=='virginica',0],color='g',alpha=.7)
xlim(xmin,xmax)
subplot(414) # global histogram (4th, on the bottom)
hist(data[:,0],color='y',alpha=.7)
xlim(xmin,xmax)
show()


# https://blog.csdn.net/suibianshen2012/article/details/51880778?utm_source=blogxgwz1