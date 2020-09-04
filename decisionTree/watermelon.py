import numpy as np


def getData():
    file = 'watermelon2_data.txt'

    labels = []
    f = open(file, 'r', encoding='utf-8')
    arrayOlines = f.readlines()
    numberOfLines = len(arrayOlines)
    print(arrayOlines[0])
    featureNum = len(arrayOlines[0].split(',')) - 2
    returnData = np.zeros((numberOfLines - 1, featureNum))
    print(numberOfLines)
    for i in range(numberOfLines - 1):
        row = arrayOlines[i + 1].split(',')
        returnData[i,:] = row[1:-1]
        labels.append(row[-1])
    return returnData,labels
returnData,labels = getData()
print(returnData)
print(labels)
