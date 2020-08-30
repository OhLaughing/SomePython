import warnings
from os import listdir
from unittest import TestCase

import knn.handwriting_kdTree as hand
import knn.kdTest as kd
from knn.kdTest import *


class Test(TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', ResourceWarning)

    def test_init_kd_tree(self):
        path = r'F:\workspace\pdf\机器学习实战\Machine-Learning-in-Action-Python3-master\kNN_Project2'
        trainData, trainLabels = hand.getTrainData(path + '\\trainingDigits')
        T = np.c_[trainData, np.arange(trainData.shape[0])]
        tree = kd.initKdTree(T, 1, None, 0)

        testFileList = listdir(path + '\\testDigits')
        # 错误检测计数
        errorCount = 0.0
        # 测试数据的数量
        mTest = len(testFileList)
        k = 3
        for i in range(mTest):
            fileNameStr = testFileList[i]
            classNumber = int(fileNameStr.split('_')[0])
            # 获得测试集的1*1024向量，用于训练
            vectorUnderTest = utils.img2vector(path + '/testDigits/%s' % (fileNameStr))

            sortedDist = utils.getSortedDist(vectorUnderTest, trainData, k)
            distant1 = utils.getDistants(trainData, sortedDist, vectorUnderTest)

            nearestkNodeList = []
            kd.findkNearestNode(nearestkNodeList, tree, vectorUnderTest, 3)

            list = utils.getList(nearestkNodeList)
            list = list[::-1]

            distant2 = utils.getDistants(trainData, list, vectorUnderTest)
            print(list)
            print(distant2)
            assert (distant1 == distant2).all()
            print('*********equal : {}'.format(i))
