# # !/usr/bin/env python3
# # -*- coding:UTF-8 -*-
# # author: ChenDu
# # time: 2018/10/21 0021
#
# # from sklearn import datasets
# # from sklearn.linear_model import LinearRegression
# # import matplotlib.pyplot as plt
# #
# # #使用以后的数据集进行线性回归（这里是波士顿房价数据）
# # loaded_data=datasets.load_boston()
# # data_X=loaded_data.data
# # data_y=loaded_data.target
# #
# # model=LinearRegression()
# # model.fit(data_X,data_y)
# #
# # print(model.predict(data_X[:4,:]))
# # print(data_y[:4])
# #
# # #使用生成线性回归的数据集，最后的数据集结果用散点图表示
# # X,y=datasets.make_regression(n_samples=100,n_features=1,n_targets=1,noise=10)   #n_samples表示样本数目，n_features特征的数目  n_tragets  noise噪音
# # plt.scatter(X,y)
# # plt.show()
# # ---------------------
# # 作者：yealxxy
# # 来源：CSDN
# # 原文：https://blog.csdn.net/u014248127/article/details/78885180
# # 版权声明：本文为博主原创文章，转载请附上博文链接！
#
# # from sklearn import datasets
# #
# # # iris = datasets.load_iris()
# # iris = datasets.load_iris()
# # digits = datasets.load_digits()
# #
# # # print(digits.data)
# # # print(digits.target)
# # print(iris.data)
#
#
# # from sklearn import datasets
# # from sklearn.linear_model import LinearRegression
# # import matplotlib.pyplot as plt
# #
# # #使用以后的数据集进行线性回归（这里是波士顿房价数据）
# # loaded_data=datasets.load_boston()
# # data_X=loaded_data.data
# # data_y=loaded_data.target
# #
# # model=LinearRegression()
# # model.fit(data_X,data_y)
# #
# # print(model.predict(data_X[:4,:]))
# # print(data_y[:4])
# #
# # #使用生成线性回归的数据集，最后的数据集结果用散点图表示
# # X,y=datasets.make_regression(n_samples=100,n_features=1,n_targets=1,noise=10)   #n_samples表示样本数目，n_features特征的数目  n_tragets  noise噪音
# # plt.scatter(X,y)
# # plt.show()
# ###########################
# #（1）观察原始数据（样本）
# #知识点：数据导入；数据可视化
# ###########################
#
# ##################
# #在ubuntu15.10中通过如下6条命令来安装python环境
# #sudo apt-get install python   #安装python最新版，一般已经自带最新2.7版本了
# #sudo apt-get install python-numpy    #安装python的numpy模块
# #sudo apt-get install python-matplotlib
# #sudo apt-get install python-networkx
# #sudo apt-get install python-sklearn
# #python  #看python版本并进入交互式界面，就可以执行如下命令，全部拷贝黏贴进去试试看？
# #另外，可以下载Anaconda的Python IDE集成环境，搜一下非常好，很多SCIPY等核心库都集成了，免去安装之苦！
# #特别注意：笔者是WIN10宿主机上安装Ubuntu15.10最新虚拟机，在Ubuntu中默认安装了python，升级并安装以上lib后实践所有如下代码！
# ##################
#
# import urllib2
# url = 'http://aima.cs.berkeley.edu/data/iris.csv'
# u = urllib2.urlopen(url)
# #以下为本地样本存储路径，请根据实际情况设定！
# #localfn='/mnt/hgfs/sharedfolder/iris.csv' #for linux
# localfn='D:\\Virtual Machines\\sharedfolder\\iris.csv' #for windows
# localf = open(localfn, 'w')
# localf.write(u.read())
# localf.close()
#
# # data examples
# #COL1,  COL2,   COL3,   COL4,   COL5
# #5.1    3.5    1.4    0.2    setosa
# #…    …    …    …    …
# #4.7    3.2    1.3    0.2    setosa
# #7    3.2    4.7    1.4    versicolor
# #…    …    …    …    …
# #6.9    3.1    4.9    1.5    versicolor
# #6.3    3.3    6    2.5    virginica
# #…    …    …    …    …
# #7.1    3    5.9    2.1    virginica
#
# #############################
# #U can get description of 'iris.csv'
# #at 'http://aima.cs.berkeley.edu/data/iris.txt'
# #Definiation of COLs:
# #1. sepal length in cm (花萼长)
# #2. sepal width in cm（花萼宽）
# #3. petal length in cm (花瓣长)
# #4. petal width in cm（花瓣宽）
# #5. class:
# #      -- Iris Setosa
# #      -- Iris Versicolour
# #      -- Iris Virginica
# #Missing Attribute Values: None
# #################################
#
#
# from numpy import genfromtxt, zeros
# # read the first 4 columns
# data = genfromtxt(localfn,delimiter=',',usecols=(0,1,2,3))
# # read the fifth column
# target = genfromtxt(localfn,delimiter=',',usecols=(4),dtype=str)
#
# print(data.shape)
# # output: (150, 4)
# print(target.shape)
# # output: (150,)
#
# #auto build a collection of unique elements
# print(set(target))
# # output: set(['setosa', 'versicolor', 'virginica'])
# #print set(data) #wrong usage of set, numbers is unhashable
#
# ######################
# #plot库用法简述：
# #'bo'=blue+circle; 'r+'=red+plus;'g'=red+*
# #search keyword 'matlab plot' on web for details
# #http://www.360doc.com/content/15/0113/23/16740871_440559122.shtml
# #http://zhidao.baidu.com/link?url=6JA9-A-UT3kmslX1Ba5uTY1718Xh-OgebUJVuOs3bdzfnt4jz4XXQdAmvb7R5JYMHyRbBU0MYr-OtXPyKxnxXsPPkm9u5qAciwxIVACR8k7
# ######################
#
# #figure for 2D data
# from pylab import plot, show
# plot(data[target=='setosa',0],data[target=='setosa',2],'bo')
# plot(data[target=='versicolor',0],data[target=='versicolor',2],'r+')
# plot(data[target=='virginica',0],data[target=='virginica',2],'g*')
# show()
#
# #注意:如果在Ubuntu的python交互式环境下运行，则figure会打断程序的RUN.
# #如果你用Anaconda的spyder（Python2.7）则方便的多，生成的figure会自动输出到console
# #且不会打断程序运行！
#
# #figure for all 4D（4个维度） data, 同色一类，圈是花萼，加号花瓣
# setosa_sepal_x=ssx=data[target=='setosa',0]
# setosa_sepal_y=ssy=data[target=='setosa',1]
# setosa_petal_x=spx=data[target=='setosa',2]
# setosa_petal_y=spy=data[target=='setosa',3]
#
# versicolor_sepal_x=vsx=data[target=='versicolor',0]
# versicolor_sepal_y=vsy=data[target=='versicolor',1]
# versicolor_petal_x=vpx=data[target=='versicolor',2]
# versicolor_petal_y=vpy=data[target=='versicolor',3]
#
# virginica_sepal_x=vgsx=data[target=='virginica',0]
# virginica_sepal_y=vgsy=data[target=='virginica',1]
# virginica_petal_x=vgpx=data[target=='virginica',2]
# virginica_petal_y=vgpy=data[target=='virginica',3]
#
# plot(ssx,ssy,'bo',spx,spy,'b+')
# plot(vsx,vsy,'ro',vpx,vpy,'r+')
# plot(vgsx,vgsy,'go',vgpx,vgpy,'g+')
# show()
#
#
# #figure for 1D（花萼的长度），三类长度及平均值的直方图
# #pylab详细用法参考如下
# #http://hyry.dip.jp/tech/book/page/scipy/matplotlib_fast_plot.html
# from pylab import figure, subplot, hist, xlim, show
# xmin = min(data[:,0])
# xmax = max(data[:,0])
# figure() #可省略，默认会生成一个figure
# subplot(411) # distribution of the setosa class (1st, on the top)
# hist(data[target=='setosa',0],color='b',alpha=.7)
# xlim(xmin,xmax)
# #subplot（行,列,plot号）；(4,1,2)合并为412,都小于10可合成
# subplot(412) # distribution of the versicolor class (2nd)
# hist(data[target=='versicolor',0],color='r',alpha=.7)
# xlim(xmin,xmax)
# subplot(413) # distribution of the virginica class (3rd)
# hist(data[target=='virginica',0],color='g',alpha=.7)
# xlim(xmin,xmax)
# subplot(414) # global histogram (4th, on the bottom)
# hist(data[:,0],color='y',alpha=.7)
# xlim(xmin,xmax)
# show()
#
# # https://blog.csdn.net/suibianshen2012/article/details/51880778?utm_source=blogxgwz1

from collections import Counter, defaultdict
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

matplotlib.rcParams['font.sans-serif'] = ['SimHei']

iris_datas = datasets.load_iris()
style_list = ['o', '^', 's']  # 设置点的不同形状，不同形状默认颜色不同，也可自定义
data = iris_datas.data
labels = iris_datas.target_names
cc = defaultdict(list)[]

for i, d in enumerate(data):
    cc[labels[int(i / 50)]].append(d)

p_list = []
c_list = []

for each in [0, 2]:
    for i, (c, ds) in enumerate(cc.items()):
        draw_data = np.array(ds)
        p = plt.plot(draw_data[:, each], draw_data[:, each + 1], style_list[i])
        p_list.append(p)
        c_list.append(c)

    plt.legend(map(lambda x: x[0], p_list), c_list)
    plt.title('鸢尾花花瓣的长度和宽度') if each else plt.title('鸢尾花花萼的长度和宽度')
    plt.xlabel('花瓣的长度(cm)') if each else plt.xlabel('花萼的长度(cm)')
    plt.ylabel('花瓣的宽度(cm)') if each else plt.ylabel('花萼的宽度(cm)')
    plt.show()