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

class Test(TestCase):
    def test_init_kd_tree(self):
        path = r'D:\workspace\pdfd\machineLeanringInAction\Machine-Learning-in-Action-Python3-master\kNN_Project2'
        trainData, trainLabels = hand.getTrainData(path + '\\trainingDigits')
        T = np.c_[trainData, np.arange(trainData.shape[0])]
        tree = kd.initKdTree(T, 1, None, 0)

        testFileList = listdir(path + '\\testDigits')
        # 错误检测计数
        errorCount = 0.0
        # 测试数据的数量
        mTest = len(testFileList)
        for i in range(1):
            fileNameStr = testFileList[i]
            classNumber = int(fileNameStr.split('_')[0])
            # 获得测试集的1*1024向量，用于训练
            vectorUnderTest = hand.img2vector(path + '/testDigits/%s' % (fileNameStr))

            sortedDist = getSortedDist(vectorUnderTest, trainData)[0:3]

            knearestList = kd.findkNearestNode(tree, vectorUnderTest, 3)

            for j in range(3):
                assert sortedDist[j] == knearestList[j].node.point

            print(sortedDist)
