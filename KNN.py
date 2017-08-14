#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
KNN算法基本步骤：
1.计算已知类别数据集中的点与当前点之间的距离
2.按照距离递增次序排序
3.选取与当前点距离最小的k个点
4.确定前k个点所在类别的出现频率
5.返回前k个点出现频率最高的类别作为当前点的预测分类
"""

import numpy as np
import operator
import pdb


class KNN(object):

    def __init__(self, k, inX):
        self.k = k  # KNN算法中的k值(正整数）
        self.group = np.array  # 训练样本数据集，格式为数组
        self.labels = []  # 标签
        self.inX = inX  # 用于分类的输入向量

    # 训练样本
    def createDataSet(self):
        self.group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
        self.labels = ["A", "A", "B", "B"]

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

    tit = KNN(k=3, inX=[0, 0])
    tit.createDataSet()
    result = tit.classifyO()
    print result

    print "---done---"