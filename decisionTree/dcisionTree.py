import math
from decisionTree.watermelon import getData
import numpy as np

def getEnt(data):
    ent = 0.0
    totalnum =0
    for key, num in data.items():
        totalnum+=num
    for key, num in data.items():
        ent += num * math.log(num / totalnum, 2)
    return -ent/totalnum, totalnum

def entropy(labels, indexList):
    total = len(indexList)
    classification = {}
    for i in indexList:
        if labels[i] in classification:
            classification[labels[i]] += 1.0
        else:
            classification[labels[i]] = 1.0

    ent = getEnt(classification)
    return ent

def put(total, data, label):
    d = total[data]
    if label in d:
        d[label]+=1
    else:
        d[label]=1

# 计算各个属性的信息熵
def information_entropy(properties, labels, indexList):
    total ={}
    for i in indexList:
        if properties[i] not in total:
            total[properties[i]] = {}
        put(total, properties[i], labels[i])

    print(total)

    each_ent = []
    each_num =[]
    for key, value in total.items():
        ent, num = getEnt(value)
        each_ent.append(ent)
        each_num.append(num)

    return each_ent, each_num


def information_gain(ent, each_ent, each_num):
    sum1 = sum(each_num)
    s = 0.0
    for i in range(len(each_num)):
        s+= (each_num[i]/sum1)*each_ent[i]
    return ent - s



if __name__ == '__main__':
    data, labels = getData()
    print(data)
    print(labels)
    indexList = np.arange(len(labels))
    ent, totalNum = entropy(labels,indexList)
    print(ent)
    info_gains=[]
    for i in range(data.shape[1]):
        each_ent, each_num = information_entropy(data[:,i], labels, indexList)
        print(each_ent)
        print(each_num)
        print(sum(each_num))
        info_gain = information_gain(ent, each_ent, each_num)
        info_gains.append(info_gain)
        print(info_gain)
    print(info_gains)
