from decisionTree.dcisionTree import *
from decisionTree.DecisionTree import *

def getLensesData(file, spliter):
    f = open(file, 'r', encoding='utf-8')
    arrayOlines = f.readlines()
    numberOfLines = len(arrayOlines)
    features = arrayOlines[0].strip().split(spliter)[1:]
    featureNum = len(features)
    returnData = np.zeros((numberOfLines, featureNum))

    allRelations = []  # 用于记录中文属性和数字的对应关系
    for i in range(featureNum):
        allRelations.append({})
    print(numberOfLines)
    for i in range(numberOfLines - 1):
        row = arrayOlines[i].strip().split(spliter)
        rowData = handlerRowData(allRelations, row[1:])
        returnData[i, :] = rowData
    labels = returnData[:, -1]
    returnData = returnData[:, 0:-1]
    return returnData, labels, allRelations


def handlerRowData(allRelations, rowData):
    l = len(rowData)
    row = np.zeros(l)
    for i in range(l):
        curr = rowData[i]
        if curr not in allRelations[i]:
            allRelations[i][curr] = len(allRelations[i])
        row[i] = allRelations[i][curr]
    return row


if __name__ == '__main__':

    returnData, labels, allRelations = getLensesData('lenses.txt', '\t')
    allRelations = getAllRelations(allRelations)
    features = ['age', 'aa', 'reduced?']
    print(returnData)
    print(labels)
    print(allRelations)

    tree = initDecisionTree(returnData, labels, list(range(len(features))), list(range(len(labels))))
    print(tree)

    createPlot(tree, allRelations,features)
