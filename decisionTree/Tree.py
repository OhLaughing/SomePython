from decisionTree.dcisionTree import *
from decisionTree.watermelon import getData


class Tree:
    def __init__(self, data, labels, attrList, indexList):
        self.data = data
        self.labels = labels
        self.attrList = attrList
        self.indexList = indexList
        self.label = None
        self.parent = None


# 选择最优划分属性
def chooseBestAttr(data, labels, attrList, indexList):
    ent, total = entropy(labels, indexList)
    info_gains = []
    for index in attrList:
        each_ent, each_num = information_entropy(data[:, index], labels, indexList)
        print(each_ent)
        print(each_num)
        print(sum(each_num))
        info_gain = information_gain(ent, each_ent, each_num)
        info_gains.append(info_gain)
        print(info_gain)
    biggest = info_gains[0]
    bigIndex = 0
    for i in range(len(info_gains)):
        if info_gains[i] > biggest:
            biggest = info_gains[i]
            bigIndex = i
    return bigIndex


def getDataDevideByAttrValue(attrData, indexList):
    attrCategory = {}
    for index in indexList:
        if attrData[index] in attrCategory:
            attrCategory[attrData[index]].append(index)
        else:
            t = []
            t.append(index)
            attrCategory[attrData[index]] = t

    return attrCategory


def generateTree(data, labels, features, attrList, indexList):
    theLabels = labels[indexList]
    currNode = Tree(data, labels, indexList)
    # D中样本全属于同一类别C
    if (sum(labels) == 1):
        currNode.label = labels[0]
        return currNode
    # A=空 or D中样本在A上取值相同
    elif len(attrList) == 0 or sum(theLabels) == 1:
        pass

    bigIndex = chooseBestAttr(data, labels, attrList, indexList)
    nextAttrList = attrList.copy()
    nextAttrList.remove(bigIndex)
    print('bigestAttr: ' + features[bigIndex])

    #     将该属性的所有取值的样本进行划分
    dataOfAttr = getDataDevideByAttrValue(data[:, bigIndex], indexList)
    for attr, indexs in dataOfAttr.items():
        node = Tree(data,labels, nextAttrList, indexs)
        
    print(dataOfAttr)

if __name__ == '__main__':
    data, labels, features = getData()
    attrList = np.arange(data.shape[1])
    indexList = np.arange(data.shape[0])
    head = generateTree(data, labels, features, attrList, indexList)
