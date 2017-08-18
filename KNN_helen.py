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
import pylab
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import matplotlib
import pdb

import numpy as np
import matplotlib.pyplot as plt


class KNNhelen(object):

    def __init__(self, k, inX, filename, hoRatio):
        self.k = k  # KNN算法中的k值(正整数）
        self.inX = inX  # 用于分类的输入向量
        self.filename = filename  # 训练样本的文件名字符串
        self.hoRatio = hoRatio  # 测试数据的大小，通常采用90%的数据作为训练样本训练分类器，其余的10%的数据作为测试样本去测试分类器

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

    # 图形化展示样本数据
    def show(self):
        fig = plt.figure(figsize=(12, 6))  # 创建一幅图，指定图的尺寸。
        pylab.mpl.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体
        # mpl.rcParams['axes.unicode_minus'] = False  # 支持负号显示
        plt.suptitle(u'训练样本散点图')  # 整个图的大标题
        p1 = fig.add_subplot(1, 2, 1)  # 将图分割成1行2列，图像画在从左到右从上到下的第1块
        p1.set_title(u'图1')  # 子图标题
        p1.set_xlabel(u'玩视频游戏所耗时间百分比')  # 子图x横坐标标注
        p1.set_ylabel(u'每周所消费的冰淇淋公升数')  # 子图y纵坐标标注
        T = np.arctan2(self.returnMat[:, 1], self.returnMat[:, 2])  # 计算x除以y的商的arctan的角度（弧度）。c参数指定每个点的颜色，可以是数值或数组。这里使用一维数组为每个点指定了一个数值。通过颜色映射表，每个数值都会与一个颜色相对应。默认的颜色映射表中蓝色与最小值对应，红色与最大值对应。当c参数是形状为(N,3)或(N,4)的二维数组时，则直接表示每个点的RGB颜色。
        p1.scatter(self.returnMat[:, 1], self.returnMat[:, 2], c=T, s=25, alpha=0.4, marker='o')  # 绘制散点图，x横坐标为returnMat特征矩阵的第二列特征数据（玩视频游戏所耗时间百分比），y纵坐标为returnMat特征矩阵的第三列特征数据（每周所消费的冰淇淋公升数）。参数C表示散点颜色，参数s表示散点大小，参数alpha表示透明度，参数marker表示散点形状。
        p2 = fig.add_subplot(1, 2, 2)  # 将图分割成1行2列，图像画在从左到右从上到下的第2块
        p2.set_title(u'图2')
        p2.set_xlabel(u'每年获得的飞行常客里程数')
        p2.set_ylabel(u'玩视频游戏所耗时间百分比')
        p2.scatter(self.returnMat[:, 0], self.returnMat[:, 1], 15.0*np.array(self.classLabelVectornum), 15.0*np.array(self.classLabelVectornum))  # 利用数字化的类标签向量指定颜色参数c和尺寸参数s，按类别个性化的指定每一个点的颜色和尺寸。散点上绘制了色彩不等、尺寸不同的点。
        plt.show()  # 显示

    # 归一化数据
    def autoNorm(self):
        self.minVals = self.returnMat.min(0)  # 返回每一列的最小值，1*3的列数的向量
        self.maxVals = self.returnMat.max(0)  # 返回每一列的最大值，1*3的列数的向量
        self.ranges = self.maxVals - self.minVals  # 返回每一列的极差，1*3的列数的向量，即数据的取值范围
        self.normDateSet = np.zeros(self.returnMat.shape)  # 初始化特征归一化后的矩阵，维度与训练样本的特征矩阵相同
        self.normDateSet = self.returnMat - np.tile(self.minVals, (self.m, 1))  # 每一列的最小值重复m行，重复扩充成m*3的矩阵。返回特征矩阵与最小值矩阵的差
        self.normDateSet = self.normDateSet/np.tile(self.ranges, (self.m, 1))  # 每一列的极差重复m行，重复扩充成m*3的矩阵。返回特征数据归一化的矩阵
        # print self.normDateSet[0]

    # 输入向量self.inX的归一化
    def inXautoNorm(self):
        self.inX = (self.inX - self.minVals)/self.ranges
        # print self.inX

    # 分类函数，训练样本的三个特征同等重要，采用归一化后的训练样本。采用90%的数据作为训练样本训练分类器，其余的10%的数据作为测试样本去测试分类器
    def classifyO(self):
        # 计算输入向量与训练样本数据集之间的欧式距离
        #diffMat = np.tile(self.inX, (self.m, 1)) - self.normDateSet  # 归一化的训练样本数据。输入向量纵向重复m次，生成与训练样本数据集同维度的矩阵，再矩阵相减得到差
        #diffMat = np.tile(self.inX, (self.m, 1)) - self.returnMat  # 输入向量纵向重复m次，生成与训练样本数据集同维度的矩阵，再矩阵相减得到差

        numTestVecs = int(self.m * self.hoRatio)  # 测试样本数量
        numDateVecs = self.m - numTestVecs  #训练样本数量
        #diffMat = np.tile(self.inX, (numDateVecs, 1)) - self.returnMat[numTestVecs:self.m, :]  # 选取一定数量的训练样本。输入向量纵向重复m次，生成与训练样本数据集同维度的矩阵，再矩阵相减得到差
        diffMat = np.tile(self.inX, (numDateVecs, 1)) - self.normDateSet[numTestVecs:self.m, :]  # 选取一定数量的归一化后的训练样本。输入向量纵向重复m次，生成与训练样本数据集同维度的矩阵，再矩阵相减得到差

        sqDiffMat = diffMat ** 2  # 矩阵中的元素求平方
        sqDistances = np.sum(sqDiffMat, axis=1)  # 矩阵按照行的方向求和
        distances = sqDistances ** 0.5  # 矩阵中的元素开根号

        # 距离按升序排序
        sortedDistIndicies = np.argsort(distances)  # 数组升序排列，返回数组值从小到大的索引值
        #print len(sortedDistIndicies)

        # 选取距离最小的前k个标签，并计算相应标签的频数
        classCount = {}
        for i in range(self.k):
            voteIlabel = self.classLabelVector[sortedDistIndicies[i]+numTestVecs]  # 取前k个距离最小的样本的标签
            classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 计算标签的频数

        # 对距离最小的前k个标签的频数进行降序排序，返回频数最大时对应的标签
        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)  #对classCount中的频数进行降序排序
        return sortedClassCount[0][0]  # 返回频数最大时，对应的标签



if __name__ == '__main__':
    tit = KNNhelen(filename='datingTestSet.txt', k=3, inX=[1000, 10, 0.5], hoRatio=0.1)
    tit.file2matrix()
    #tit.show()
    tit.autoNorm()
    tit.inXautoNorm()
    result = tit.classifyO()
    print result

    print "---done---"