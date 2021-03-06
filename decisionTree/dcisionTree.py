import math

from decisionTree.DecisionTree import createPlot
from decisionTree.watermelon import *


def getEnt(data, totalnum):
    ent = 0.0
    for key, num in data.items():
        ent += num * math.log(num / totalnum, 2)
    return -ent / totalnum


def entropy(labels, indexList):
    total = len(indexList)
    classification = {}
    for i in indexList:
        if labels[i] in classification:
            classification[labels[i]] += 1.0
        else:
            classification[labels[i]] = 1.0
    print("indexList: {}".format(indexList))

    ent = getEnt(classification, len(indexList))
    return ent


def put(total, data, label):
    d = total[data]
    if label in d:
        d[label] += 1
    else:
        d[label] = 1


def getTotal(dict1):
    total = 0
    for _, num in dict1.items():
        total += num
    return total


# 计算各个属性的信息熵
def information_entropy(properties, labels, indexList):
    total = {}
    for i in indexList:
        if properties[i] not in total:
            total[properties[i]] = {}
        put(total, properties[i], labels[i])

    print(total)

    each_ent = []
    each_num = []
    for key, value in total.items():
        total = getTotal(value)
        ent = getEnt(value, total)
        each_ent.append(ent)
        each_num.append(total)

    return each_ent, each_num

# ID3决策树采用信息增益来选择最优属性
def information_gain(ent, each_ent, each_num):
    sum1 = sum(each_num)
    s = 0.0
    for i in range(len(each_num)):
        s += (each_num[i] / sum1) * each_ent[i]
    return ent - s


def getBestAttr(info_gains):
    bigest = -1.0
    index = -1
    for k, v in info_gains.items():
        if v > bigest:
            bigest = v
            index = k
    return index


def getSubIndex(theAttrList, dataIndexList):
    t = {}
    for i in dataIndexList:
        if theAttrList[i] in t:
            t[theAttrList[i]].append(i)
        else:
            t[theAttrList[i]] = [i]
    return t


def otherAttrList(attrIndexList, theIndex):
    newIndex = []
    for i in attrIndexList:
        if theIndex != i:
            newIndex.append(i)
    return newIndex


def getValueNum(labels, indexList):
    numDict = {}
    for i in indexList:
        if labels[i] in numDict:
            numDict[int(labels[i])] += 1
        else:
            numDict[int(labels[i])] = 1
    return numDict


def getLargestNum(numDict):
    index = 0
    largest = 0
    for k, num in numDict.items():
        if num > largest:
            largest = num
            index = k
    return index


def initDecisionTree(data, labels, featureList, dataIndexList):
    print("attrIndexList: {}".format(featureList))
    print("dataIndexList: {}".format(dataIndexList))
    numDict = getValueNum(labels, dataIndexList)
    if len(numDict) == 1:
        return labels[dataIndexList[0]]
    if len(featureList) == 1:
        return getLargestNum(numDict)
    ent = entropy(labels, dataIndexList)
    info_gains = {}
    for i in featureList:
        each_ent, each_num = information_entropy(data[:, i], labels, dataIndexList)
        info_gain = information_gain(ent, each_ent, each_num)
        info_gains[i] = info_gain
    print(info_gains)
    bestAttr = getBestAttr(info_gains)
    otherAttrs = otherAttrList(featureList, bestAttr)
    subIndex = getSubIndex(data[:, bestAttr], dataIndexList)
    nextTree = {}

    for value, index in subIndex.items():
        # if len(index) > 0:
        nextTree[value] = initDecisionTree(data, labels, otherAttrs, index)
    return {int(bestAttr): nextTree}


def getAllRelations(relations):
    returnRelations = []
    for i in range(len(relations)):
        a = relations[i]
        tmpDict = {}
        for k, v in a.items():
            tmpDict[int(v)] = k
        returnRelations.append(tmpDict)
    return returnRelations


data, labels, features = getWatermelonData()
returnData, labels1, attr_cn, allRelations = getWatermelonDigitalData()
allRelations = getAllRelations(allRelations)

if __name__ == '__main__':
    print(data)
    print(labels)
    indexList = np.arange(len(labels1))
    ent = entropy(labels1, indexList)
    print(type(allRelations[0]))
    print(ent)
    info_gains = []
    tree = initDecisionTree(returnData, labels1, list(range(len(features))), list(range(len(labels))))
    print(tree)
    createPlot(tree, allRelations, features)
