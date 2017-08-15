#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
示例1：使用KNN算法改进约会网站的配对效果
约会对象的类别：
1.不喜欢的人
2.魅力一般的人
3.极具魅力的人
训练样本datingTestSet.txt的特征(每一个样本数据占一行，共1000行)：
1.每年获得的飞行常客里程数
2.玩视频游戏所耗时间百分比
3.每周消费的冰淇淋公升数
KNN算法基本步骤：
1.计算已知类别数据集中的点与当前点之间的距离
2.按照距离递增次序排序
3.选取与当前点距离最小的k个点
4.确定前k个点所在类别的出现频率
5.返回前k个点出现频率最高的类别作为当前点的预测分类
"""

import operator
import matplotlib
import pdb

import numpy as np
import matplotlib.pyplot as plt


class KNNhelen(object):

    def __init__(self, filename):
        # self.k = k  # KNN算法中的k值(正整数）
        # self.inX = inX  # 用于分类的输入向量
        self.filename = filename  # 训练样本的文件名字符串
        self.returnmat = np.mat  # 存储训练样本中的特征数据，格式为矩阵
        self.classLabelVector = []  # 存储类标签的向量

    # 处理训练样本，输入文件名字符串，将文本类型的数据转化为矩阵和向量，输出训练样本矩阵和类标签向量
    def file2matrix(self):
        fr = open(self.filename)  # 打开文件
        arrayOLines = fr.readlines()  # 逐行读取文本文件的数据
        numberOfLines = len(arrayOLines)  # 文件中的数据总行数

        self.returnMat = np.zeros((numberOfLines, 3))  # 初始化矩阵，维度为样本数据总行数*3
        index = 0
        for line in arrayOLines:
            line = line.strip()  # 除去每一行的前后空格及换行符
            listFromLine = line.split("\t")  # 以空格分隔每一行
            self.returnMat[index, :] = listFromLine[0: 3]  # 每一行的特征数据存储到矩阵中的每一行
            self.classLabelVector.append(listFromLine[-1])  # 每一行的类标签存储到类标签向量中
            index = index + 1

        # print self.classLabelVector[0: 10]

    # 图形化展示样本数据
    def show(self):
        fig = plt.figure()  # 创建一幅图
        ax = fig.add_subplot(111)
        ax.scatter(self.returnMat[:, 1], self.returnMat[:, 2])
        plt.show()  # 显示

    # 分类函数
    def classifyO(self):
        # 计算输入向量与训练样本数据集之间的欧式距离
        dataSetSize = self.group.shape[0]  # 训练样本集的维度,取其行数
        diffMat = np.tile(self.inX, (dataSetSize, 1)) - self.group  # 输入向量纵向重复dataSetSize次，生成与训练样本数据集同维度的矩阵，再矩阵相减得到差
        sqDiffMat = diffMat ** 2  # 矩阵中的元素求平方
        sqDistances = np.sum(sqDiffMat, axis=1)  # 矩阵按照行的方向求和
        distances = sqDistances ** 0.5  # 矩阵中的元素开根号

        # 距离按升序排序
        sortedDistIndicies = np.argsort(distances)  # 数组升序排列，返回数组值从小到大的索引值

        # 选取距离最小的前k个标签，并计算相应标签的频数
        classCount = {}
        for i in range(self.k):
            voteIlabel = self.labels[sortedDistIndicies[i]]  # 取前k个距离最小的样本的标签
            classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 计算标签的频数

        # 对距离最小的前k个标签的频数进行降序排序，返回频数最大时对应的标签
        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)  #对classCount中的频数进行降序排序
        return sortedClassCount[0][0]  # 返回频数最大时，对应的标签






if __name__ == '__main__':

    tit = KNNhelen('datingTestSet.txt')
    tit.file2matrix()

    print "---done---"