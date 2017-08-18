#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
示例1：使用KNN算法改进约会网站的配对效果
测试数据集的选取：
1.通常只提供已有数据的90%作为训练样本来训练分类器，而使用其余的10%数据去测试分类器，检测分类器的正确率。
2.10%的测试数据应该是随机选择的，由于提供的数据并没有按照特定目的来排序，所以可以随意选择10%数据而不影响其随机性
"""

import numpy as np

from KNN_helen import KNNhelen


class HelenTest(object):
    def __init__(self, k, hoRatio, filename):
        self.k = k  # KNN算法中的k值(正整数）
        self.hoRatio = hoRatio  # 测试数据的大小，通常采用90%的数据作为训练样本训练分类器，其余的10%的数据作为测试样本去测试分类器
        self.filename = filename  # 训练样本的文件名字符串

        self.returnMat = np.mat  # 存储训练样本中的特征数据，格式为矩阵
        self.classLabelVector = []  # 存储类标签的向量
        self.m = int  # 存储训练样本中的特征数据矩阵的行数
        self.classLabelVectornum = []  # 存储数字化的类标签的向量
        self.minVals = np.array  # 存储训练样本中的特征数据矩阵的每一列的最小值
        self.maxVals = np.array  # 存储训练样本中的特征数据矩阵的每一列的最大值
        self.ranges = np.array  # 存储训练样本中的特征数据矩阵的每一列的极差，即数据的取值范围
        self.normDateSet = np.mat  # 存储训练样本中的归一化后的特征数据，格式为矩阵

    # 处理训练样本，输入文件名字符串，将文本类型的数据转化为矩阵和向量，输出训练样本矩阵和类标签向量
    def file2matrix(self):
        fr = open(self.filename)  # 打开文件
        arrayOLines = fr.readlines()  # 逐行读取文本文件的数据
        numberOfLines = len(arrayOLines)  # 文件中的数据总行数

        self.returnMat = np.zeros((numberOfLines, 3))  # 初始化训练样本的特征数据矩阵，维度为样本数据总行数*3
        index = 0
        for line in arrayOLines:
            line = line.strip()  # 除去每一行的前后空格及换行符
            listFromLine = line.split("\t")  # 以空格分隔每一行
            self.returnMat[index, :] = listFromLine[0: 3]  # 每一行的特征数据存储到矩阵中的每一行
            self.classLabelVector.append(listFromLine[-1])  # 每一行的类标签存储到类标签向量中
            index = index + 1

        self.m = self.returnMat.shape[0]  # 训练样本的特征矩阵的行数

        for label in self.classLabelVector:
            if label == 'didntLike':
                self.classLabelVectornum.append(1)
            elif label == 'smallDoses':
                self.classLabelVectornum.append(2)
            elif label == 'largeDoses':
                self.classLabelVectornum.append(3)
            else:
                self.classLabelVectornum.append(0)  # 异常值/缺失值的处理

                # print self.classLabelVector[0: 10]
                # print self.classLabelVectornum[0: 10]
                # print self.m

    # 归一化数据
    def autoNorm(self):
        self.minVals = self.returnMat.min(0)  # 返回每一列的最小值，1*returnMat的列数的向量
        self.maxVals = self.returnMat.max(0)  # 返回每一列的最大值，1*returnMat的列数的向量
        self.ranges = self.maxVals - self.minVals  # 返回每一列的极差，1*returnMat的列数的向量
        self.normDateSet = np.zeros(self.returnMat.shape)  # 初始化特征归一化后的矩阵，维度与训练样本的特征矩阵相同
        self.normDateSet = self.returnMat - np.tile(self.minVals,
                                                    (self.m, 1))  # 每一列的最小值重复m行，重复扩充成m*3的矩阵。返回特征矩阵与最小值矩阵的差
        self.normDateSet = self.normDateSet / np.tile(self.ranges,
                                                      (self.m, 1))  # 每一列的极差重复m行，重复扩充成m*3的矩阵。返回特征数据归一化的矩阵
        # print self.normDateSet[0]

    # 测试函数，测试分类器的分类效果。以分类器的错误率作为评价标准，错误率即分类错误的次数除以样本数据点的总数
    def datingClassTest(self):
        numTestVecs = int(self.m * self.hoRatio)  # 测试样本数量
        errorCount = 0  # 分类器分类错误的次数计数
        for i in range(numTestVecs):
            #t = KNNhelen(filename=self.filename, k=self.k, inX=self.returnMat[i, :], hoRatio=self.hoRatio)
            t = KNNhelen(filename=self.filename, k=self.k, inX=self.normDateSet[i, :], hoRatio=self.hoRatio)  # 归一化的数据集
            t.file2matrix()
            t.autoNorm()
            classifierResult = t.classifyO()
            if classifierResult != self.classLabelVector[i]:
                errorCount = errorCount + 1

        errorate = float(errorCount)/numTestVecs*100
        print "the total error rate is : %f " % errorate
        return errorate




if __name__ == '__main__':
    tit = HelenTest(k=3, hoRatio=0.1, filename='datingTestSet.txt')
    tit.file2matrix()
    tit.autoNorm()
    tit.datingClassTest()

"""
归一化       hoRatio       k          错误率（百分比）
  否          0.1          3            24.000000
  是          0.1          3             5.000000

归一化对分类器的分类效果的影响：
训练数据集的大小对分类器的分类效果的影响：
k值的选取对分类器的分类效果的影响：
"""