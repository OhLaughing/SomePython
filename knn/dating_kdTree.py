import numpy as np

import knn.kNN as knn
import knn.kdTest as kd

if __name__ == '__main__':
    filename = "datingTestSet.txt"
    # 打开并处理数据
    datingDataMat, datingLabels = knn.file2matrix(filename)
    trainlabels = datingLabels[100:]
    testlabels = datingLabels[0:100]
    # 训练集归一化
    normDataset, ranges, minVals = knn.autoNorm(datingDataMat)
    trainData = normDataset[100:, :]
    testData = normDataset[0:100, :]
    T = np.c_[trainData, np.arange(trainData.shape[0])]
    tree = kd.initKdTree(T, 1, None, 0)
    kd.printNode(tree)

    errorNum =0
    for i in range(len(testlabels)):
        nearNodeList = kd.findkNearestNode(tree, testData[i, :], 3)
        nearestLabel = kd.getNearestLabel(nearNodeList, trainlabels)
        if(nearestLabel != testlabels[i]):
            errorNum+=1

    print(errorNum)
