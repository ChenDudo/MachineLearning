# !/usr/bin/env python3
# -*- coding:UTF-8 -*-
# author: ChenDu
# time: 2018/11/10 0010


import pandas as pd
from matplotlib import pyplot as plt


iris_data=pd.read_csv("iris.csv")
iris_mean=iris_data.mean()
grouped_data=iris_data.groupby("species")
grouped_mean=grouped_data.mean()
# print(grouped_data)
# print(grouped_mean)
plt.legend(grouped_mean, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2)

# plt.show()

# iris_data.plot(kind='kde', subplots=True, figsize=(10, 6))
# plt.show()

# iris_data.plot(kind='bar',rot=45)
plt.show()