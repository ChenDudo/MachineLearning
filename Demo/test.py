# coding:UTF-8

import os
import numpy as np

# 获取当前目录
path = os.getcwd()
print(path)

# 跳至数据集路径
# data_path = os.path.join(path, "MachineLearning\DataSet\iris\iris.data")
data_path = os.path.join(path, "MachineLearning\DataSet\wine\wine.data")
print(data_path)

# 读取数据集
# with open(data_path, "r") as f:
#     print(f.read())


# 导入数据
#     input:  file_name(string):文件的存储位置
#     output: feature_data(mat):特征
#             label_data(mat):标签
#             n_class(int):类别的个数

def load_data(file_name):

    # 1、获取特征
    f = open(file_name)
    feature_data = []
    label_tmp = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split("\t")
        for i in range(len(lines) - 1):
            feature_tmp.append(float(lines[i]))
        # label_tmp.append(float(lines[-1]))
        feature_data.append(feature_tmp)
    f.close()

    # 2、获取标签
    m = len(label_tmp)
    n_class = len(set(label_tmp))  # 得到类别的个数
    label_data = np.mat(np.zeros((m, n_class)))
    for i in range(m):
        label_data[i, label_tmp[i]] = 1
    return np.mat(feature_data), label_data, n_class

if __name__ == "__main__":
    label = load_data(data_path)
    print(label)