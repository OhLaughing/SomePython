import numpy as np


def handlerRowData(allRelations, returnData, i, rowData):
    l = len(rowData)
    row = np.zeros(l)
    for i in range(l):
        curr = rowData[i]
        if curr not in allRelations[i]:
            allRelations[i][curr] = len(allRelations[i])
        row[i] = allRelations[i][curr]
    return row

def getData():
    file = 'watermelon_data2.txt'

    f = open(file, 'r', encoding='utf-8')
    arrayOlines = f.readlines()
    numberOfLines = len(arrayOlines)
    print(arrayOlines[0])
    features = arrayOlines[0].strip().split(',')[1:]
    print(features)
    featureNum = len(features)
    returnData = np.zeros((numberOfLines - 1, featureNum))

    allRelations = []  # 用于记录中文属性和数字的对应关系
    for i in range(featureNum):
        allRelations.append({})
    print(numberOfLines)
    for i in range(numberOfLines - 1):
        row = arrayOlines[i + 1].strip().split(',')
        rowData = handlerRowData(allRelations,returnData,i, row[1:])
        returnData[i, :] = rowData

    labels = returnData[:,-1]
    returnData = returnData[:,0:-1]
    return returnData, labels


returnData, labels = getData()
print(returnData)
print(labels)
