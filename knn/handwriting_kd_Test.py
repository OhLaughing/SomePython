from unittest import TestCase

from knn.kdTest import *
import knn.kdTest as kd
import knn.kNN as knn
from os import listdir
import knn.handwriting_kdTree as hand
import knn.utils as utils

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

            sortedDist = utils.getSortedDist(vectorUnderTest, trainData)[0:3]

            knearestList = kd.findkNearestNode(tree, vectorUnderTest, 3)

            for j in range(3):
                assert sortedDist[j] == knearestList[j].node.point

            print(sortedDist)
