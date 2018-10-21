# coding:UTF-8
'''
Date:20160901
@author: zhaozhiyong
'''
import numpy as np
# from lr_train import sig



def load_data(file_name):
    '''导入训练数据
    input:  file_name(string)训练数据的位置
    output: feature_data(mat)特征
            label_data(mat)标签
    '''
    f = open(file_name)  # 打开文件
    feature_data = []
    label_data = []
    for line in f.readlines():
        feature_tmp = []
        lable_tmp = []
        lines = line.strip().split("\t")
        feature_tmp.append(1)  # 偏置项
        for i in xrange(len(lines) - 1):
            feature_tmp.append(float(lines[i]))
        lable_tmp.append(float(lines[-1]))

        feature_data.append(feature_tmp)
        label_data.append(lable_tmp)
    f.close()  # 关闭文件
    return np.mat(feature_data), np.mat(label_data)


def sig(x):
    '''Sigmoid函数
    input:  x(mat):feature * w
    output: sigmoid(x)(mat):Sigmoid值
    '''
    return 1.0 / (1 + np.exp(-x))


def lr_train_bgd(feature, label, maxCycle, alpha):
    '''利用梯度下降法训练LR模型
    input:  feature(mat)特征
            label(mat)标签
            maxCycle(int)最大迭代次数
            alpha(float)学习率
    output: w(mat):权重
    '''
    n = np.shape(feature)[1]  # 特征个数
    w = np.mat(np.ones((n, 1)))  # 初始化权重
    i = 0
    while i <= maxCycle:  # 在最大迭代次数的范围内
        i += 1  # 当前的迭代次数
        h = sig(feature * w)  # 计算Sigmoid值
        err = label - h
        if i % 100 == 0:
            print
            "\t---------iter=" + str(i) + \
            " , train error rate= " + str(error_rate(h, label))
        w = w + alpha * feature.T * err  # 权重修正
    return w


def error_rate(h, label):
    '''计算当前的损失函数值
    input:  h(mat):预测值
            label(mat):实际值
    output: err/m(float):错误率
    '''
    m = np.shape(h)[0]

    sum_err = 0.0
    for i in xrange(m):
        if h[i, 0] > 0 and (1 - h[i, 0]) > 0:
            sum_err -= (label[i, 0] * np.log(h[i, 0]) + \
                        (1 - label[i, 0]) * np.log(1 - h[i, 0]))
        else:
            sum_err -= 0
    return sum_err / m


def save_model(file_name, w):
    '''保存最终的模型
    input:  file_name(string):模型保存的文件名
            w(mat):LR模型的权重
    '''
    m = np.shape(w)[0]
    f_w = open(file_name, "w")
    w_array = []
    for i in xrange(m):
        w_array.append(str(w[i, 0]))
    f_w.write("\t".join(w_array))
    f_w.close()



def load_weight(w):
    '''导入LR模型
    input:  w(string)权重所在的文件位置
    output: np.mat(w)(mat)权重的矩阵
    '''
    f = open(w)
    w = []
    for line in f.readlines():
        lines = line.strip().split("\t")
        w_tmp = []
        for x in lines:
            w_tmp.append(float(x))
        w.append(w_tmp)    
    f.close()
    return np.mat(w)

def load_data(file_name, n):
    '''导入测试数据
    input:  file_name(string)测试集的位置
            n(int)特征的个数
    output: np.mat(feature_data)(mat)测试集的特征
    '''
    f = open(file_name)
    feature_data = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split("\t")
        # print lines[2]
        if len(lines) != n - 1:
            continue
        feature_tmp.append(1)
        for x in lines:
            # print x
            feature_tmp.append(float(x))
        feature_data.append(feature_tmp)
    f.close()
    return np.mat(feature_data)

def predict(data, w):
    '''对测试数据进行预测
    input:  data(mat)测试数据的特征
            w(mat)模型的参数
    output: h(mat)最终的预测结果
    '''
    h = sig(data * w.T)#sig
    m = np.shape(h)[0]
    for i in range(m):
        if h[i, 0] < 0.5:
            h[i, 0] = 0.0
        else:
            h[i, 0] = 1.0
    return h

def save_result(file_name, result):
    '''保存最终的预测结果
    input:  file_name(string):预测结果保存的文件名
            result(mat):预测的结果
    '''
    m = np.shape(result)[0]
    #输出预测结果到文件
    tmp = []
    for i in range(m):
        tmp.append(str(result[i, 0]))
    f_result = open(file_name, "w")
    f_result.write("\t".join(tmp))
    f_result.close()    

if __name__ == "__main__":
    # 1、导入LR模型
    print("---------- 1.load model ------------")
    w = load_weight("data.txt")
    n = np.shape(w)[1]
    # 2、导入测试数据
    print("---------- 2.load data ------------")
    testData = load_data("test_data", n)
    # 3、对测试数据进行预测
    print("---------- 3.get prediction ------------")
    h = predict(testData, w)#进行预测
    # 4、保存最终的预测结果
    print("---------- 4.save prediction ------------")
    print(h)
    save_result("result", h)
    
