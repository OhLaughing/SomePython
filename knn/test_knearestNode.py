from unittest import TestCase

import knn.kNN as knn
import knn.kdTest as kd
from knn.kdTest import *

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
        normDataset = normDataset[:,:-1]
        trainData = normDataset[100:, :]
        testData = normDataset[0:100, :]

        T = np.c_[trainData, np.arange(trainData.shape[0])]
        tree = kd.initKdTree(T, 1, None, 0)
        k=3
        for i in range(100):
            theData = testData[i]

            sortedDist = utils.getSortedDist(theData, trainData,k)
            nearestkNodeList = []
            kd.findkNearestNode(nearestkNodeList,tree, theData, k)

            list = utils.getList(nearestkNodeList)
            list = list[::-1]

            distant1 = utils.getDistants(trainData, sortedDist, theData)

            print(sortedDist)
            print(distant1)
            distant2 = utils.getDistants(trainData, list, theData)
            print(list)
            print(distant2)
            assert (distant1 == distant2).all()
            print('*********equal : {}'.format(i))


