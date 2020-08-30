import numpy as np

# 用暴力排序算法得出的k近邻距离
def getSortedDist(inX, dataSet,k):
    # numpy函数shape[0]返回dataSet的行数
    dataSetSize = dataSet.shape[0]
    # 将inX重复dataSetSize次并排成一列
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # 二维特征相减后平方（用diffMat的转置乘diffMat）
    sqDiffMat = diffMat ** 2
    # sum()所有元素相加，sum(0)列相加，sum(1)行相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方，计算出距离
    distances = sqDistances ** 0.5
    # argsort函数返回的是distances值从小到大的--索引值
    sortedDistIndicies = distances.argsort()
    # 定义一个记录类别次数的字典
    sorted = np.array(sortedDistIndicies[0:k])

    return sorted

def getList(knearestList):
    list = np.zeros_like(knearestList)
    for i in range(len(knearestList)):
        list[i] = knearestList[i].node.index
    return list

def img2vector(filename):
    # 创建1*1024零向量
    returnVect = np.zeros((1, 1024))
    # 打开文件
    fr = open(filename)
    # 按行读取
    for i in range(32):
        # 读取一行数据
        lineStr = fr.readline()
        # 每一行的前32个数据依次存储到returnVect中
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    # 返回转换后的1*1024向量
    return returnVect[0,:]

def getDistant(x, y):
    dif = x - y
    square = dif ** 2
    sum = np.sum(square)
    return sum ** 0.5


def getDistants(trainData, list, x):
    distant = np.zeros(len(list))
    for i in range(len(list)):
        a = list[i]
        c = trainData[a]
        b = getDistant(c, x)
        distant[i] = b
    return distant