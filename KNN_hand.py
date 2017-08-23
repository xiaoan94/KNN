#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
示例2：手写识别系统
图像数据以二进制的形式存储，以文本的形式存储图像数据，虽不能有效利用内存，但方便理解。
系统只能识别数字0-9，需要识别的数字已经用图形处理软件处理成具有相同的色彩和大小，即32*32像素的黑白图像。
训练样本trainingDigits：一共大约2000个例子，每个数字大约有200个样本
测试样本testDigits：一共大约900个例子，每个数字大约有90个样本
两组数据没有覆盖
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

from os import listdir

class KNN(object):

    def __init__(self, k, filename1, filename2):
        self.k = k  # KNN算法中的k值(正整数）
        self.filename1 = filename1
        self.filename2 = filename2
        self.trainingMat = np.array  # 存储处理后的训练样本数据
        self.hwLabels = []  # 存储训练样本的标签
        self.vectorUnderTest = np.array  # 存储处理后的测试样本数据
        self.hwLabelsTest = []  # 存储测试样本的标签
        self.mTest = int  # 测试样本数量

    # 训练样本，将一个图像格式化处理为一个向量，即把一个32*32的二进制图像矩阵转换为1*1024的向量
    def img2vector(self):
        trainingFileList = listdir(self.filename1)  # 列出该目录下的文件名，列表中的文件名无序排列
        m = len(trainingFileList)  # 训练样本数量
        self.trainingMat = np.zeros((m, 1024))   #初始化训练样本矩阵，一行代表一个图像
        for i in range(m):
            fileNameStr = trainingFileList[i]
            fileStr = fileNameStr.split(".")[0]  # 去掉文件名后面的.txt的后缀
            classNumStr = int(fileStr.split("_")[0])  # 取'_'前面的分类数字标签
            self.hwLabels.append(classNumStr)
            fr = open("trainingDigits/%s" % fileNameStr)
            lineStrs = fr.readlines()
            for j in range(32):  # 循环读取文件中的前32行
                lineStr = lineStrs[j]
                for k in range(32):  # 循环读取文件中的前32个字符
                    self.trainingMat[i, 32*j+k] = int(lineStr[k])

        #print self.trainingMat[2, 0:31]
        #print self.hwLabels[0: 10]

    # 测试样本
    def img2vectorTest(self):
        testFileList = listdir(self.filename2)
        self.mTest = len(testFileList)  # 测试样本数量
        self.vectorUnderTest = np.zeros((self.mTest, 1024))
        for i in range(self.mTest):
            fileNameStr = testFileList[i]
            fileStr = fileNameStr.split(".")[0]
            classNumStr = int(fileStr.split("_")[0])
            self.hwLabelsTest.append(classNumStr)
            fr = open("testDigits/%s" % fileNameStr)
            lineStrs = fr.readlines()
            for j in range(32):
                lineStr = lineStrs[j]
                for k in range(32):
                    self.vectorUnderTest[i, 32*j+k] = int(lineStr[k])

        #print self.vectorUnderTest[13, 0:31]

    # 分类函数
    def classifyO(self, inX):
        # 计算输入向量与训练样本数据集之间的欧式距离
        dataSetSize = self.trainingMat.shape[0]  # 训练样本集的维度,取其行数
        diffMat = np.tile(inX, (dataSetSize, 1)) - self.trainingMat  # 输入向量纵向重复dataSetSize次，生成与训练样本数据集同维度的矩阵，再矩阵相减得到差
        sqDiffMat = diffMat ** 2  # 矩阵中的元素求平方
        sqDistances = np.sum(sqDiffMat, axis=1)  # 矩阵按照行的方向求和
        distances = sqDistances ** 0.5  # 矩阵中的元素开根号

        # 距离按升序排序
        sortedDistIndicies = np.argsort(distances)  # 数组升序排列，返回数组值从小到大的索引值

        # 选取距离最小的前k个标签，并计算相应标签的频数
        classCount = {}
        for i in range(self.k):
            voteIlabel = self.hwLabels[sortedDistIndicies[i]]  # 取前k个距离最小的样本的标签
            classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 计算标签的频数

        # 对距离最小的前k个标签的频数进行降序排序，返回频数最大时对应的标签
        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)  #对classCount中的频数进行降序排序
        return sortedClassCount[0][0]  # 返回频数最大时，对应的标签

    # 测试函数
    def handwritingClassTest(self):
        tit.img2vector()  # 处理后的训练样本
        tit.img2vectorTest()  # 处理后的测试样本
        errorCount = 0
        for i in range(self.mTest):
            classifierResult = self.classifyO(self.vectorUnderTest[i, :])
            if classifierResult != self.hwLabelsTest[i]:
                errorCount = errorCount + 1

        errorate = float(errorCount) / self.mTest * 100
        print "the total error is : %d " % errorCount
        print "the total error rate is : %f " % errorate
        return errorate






if __name__ == '__main__':

    tit = KNN(k=3, filename1="trainingDigits", filename2="testDigits")
    #tit.img2vector()
    #tit.img2vectorTest()
    result = tit.handwritingClassTest()
    #print result

    print "---done---"