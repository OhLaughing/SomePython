from os import listdir

import numpy as np

import knn.kdTest as kd
import knn.utils as utils

def getTrainData(path):
    hwLabels = []
    # 返回trainingDigits目录下的文件名
    trainingFilesList = listdir(path)
    # 返回文件夹下文件的个数
    m = len(trainingFilesList)
    # 初始化训练的Mat矩阵（全零阵），测试集
    trainingMat = np.zeros((m, 1024))
    # 从文件名中解析出训练集的类别
    for i in range(m):
        # 获得文件的名字
        fileNameStr = trainingFilesList[i]
        # 获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        # 将获得的类别添加到hwLabels中
        hwLabels.append(classNumber)
        # 将每一个文件的1x1024数据存储到trainingMat矩阵中
        trainingMat[i, :] = utils.img2vector(path + '/%s' % (fileNameStr))
    return trainingMat, hwLabels


if __name__ == '__main__':
    path = r'F:\workspace\pdf\机器学习实战\Machine-Learning-in-Action-Python3-master\kNN_Project2'
    trainData, trainLabels = getTrainData(path + '\\trainingDigits')
    T = np.c_[trainData, np.arange(trainData.shape[0])]
    tree = kd.initKdTree(T, 1, None, 0)

    testFileList = listdir(path + '\\testDigits')
    # 错误检测计数
    errorCount = 0.0
    # 测试数据的数量
    mTest = len(testFileList)
    # 从文件中解析出测试集的类别并进行分类测试
    total =0
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        # 获得测试集的1*1024向量，用于训练
        vectorUnderTest = utils.img2vector(path + '/testDigits/%s' % (fileNameStr))

        nearestkNodeList = []
        kd.findkNearestNode(nearestkNodeList,tree, vectorUnderTest, 4)
        theLabel = kd.getNearestLabel(nearestkNodeList, trainLabels)

        if (classNumber != theLabel):
            errorCount += 1
        total +=1
    print(str(errorCount) +" total: " +  str(total))
