# !/usr/bin/env python3
# -*- coding:UTF-8 -*-
# author: ChenDu
# time: 2018/10/28 0028

import pandas as pd
from matplotlib import pyplot as plt

iris_data = pd.read_csv("iris.csv")

grouped_data = iris_data.groupby('species')
print(grouped_data)

# group_mean = grouped_data.mean()
#
# group_mean.plot(kind="bar")
# plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2)
# plt.show()