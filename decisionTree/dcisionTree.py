import math
from decisionTree.watermelon import getData
import numpy as np
def entropy(labels, indexList):
    total = len(indexList)
    classification = {}
    for i in indexList:
        if labels[i] in classification:
            classification[labels[i]] += 1.0
        else:
            classification[labels[i]] = 1.0

    ent = 0.0
    for key, num in classification.items():
        ent += num*math.log(2, float(num / total))

    return -ent/total

# 计算各个属性的信息熵
def information_entropy(properties, labels, indexList):
    total ={}
    for i in indexList:
        if properties[i] in total:

        else:

    # pass


if __name__ == '__main__':
    data, labels = getData()
    print(data)
    print(labels)
    indexList = np.arange(len(labels))
    ent = entropy(labels,indexList)
    print(ent)
