from unittest import TestCase

from knn.kdTest import *
import knn.kdTest as kd
import knn.kNN as knn
from os import listdir
import knn.handwriting_kdTree as hand

def getSortedDist(inX, dataSet):
    # numpy函数shape[0]返回dataSet的行数
    dataSetSize = dataSet.shape[0]
    # 将inX重复dataSetSize次并排成一列
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # 二维特征相减后平方（用diffMat的转置乘diffMat）
    sqDiffMat = diffMat**2
    # sum()所有元素相加，sum(0)列相加，sum(1)行相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方，计算出距离
    distances = sqDistances**0.5
    # argsort函数返回的是distances值从小到大的--索引值
    sortedDistIndicies = distances.argsort()
    # 定义一个记录类别次数的字典
    return sortedDistIndicies

'''
    测试自己实现的kd数算法获取的k近临点是否与暴力算法获取的k近临点一致
'''
class Test(TestCase):
    def test_init_kd_tree(self):
        filename = "datingTestSet.txt"
        datingDataMat, datingLabels = knn.file2matrix(filename)
        trainlabels = datingLabels[100:]
        testlabels = datingLabels[0:100]
        # 训练集归一化
        normDataset, ranges, minVals = knn.autoNorm(datingDataMat)
        trainData = normDataset[100:, :]
        testData = normDataset[0:100, :]

        T = np.c_[trainData, np.arange(trainData.shape[0])]
        tree = kd.initKdTree(T, 1, None, 0)

        for i in range(100):
            theData = testData[i]

            sortedDist = getSortedDist(theData, trainData)[0:3]

            knearestList = kd.findkNearestNode(tree, theData, 3)

            print(sortedDist)
            print(knearestList)


