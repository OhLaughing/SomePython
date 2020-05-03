import numpy as np
import xlrd


# 读取excel中的64个测试数据
def loadTrainData():
    totalData = np.empty((64, 12))
    totallabel = np.empty(64)
    data = xlrd.open_workbook('附录A.xlsx')
    table = data.sheet_by_name('Data')
    for i in range(64):
        first = 10 + 4 * i
        data, label = readOneData(table, first)
        totalData[i] = data
        totallabel[i] = label
    return totalData, totallabel


# 读取excel中的64个测试数据
def loadCNNTrainData(num):
    datas = np.empty((num, 6, 6))
    labels = np.empty((num, 3))
    data = xlrd.open_workbook('附录B.xlsx')
    table = data.sheet_by_name('Data')
    for i in range(num):
        oneData, onelabel = readOneData6_6(table, 11 + 6 * i)
        datas[i] = oneData
        labels[i] = onelabel
    return datas, labels

def readOneData6_6(table, first):
    data = np.empty((6, 6))
    for i in range(6):
        row = table.row_values(2 + i)
        for j in range(6):
            data[i, j] = float(row[first + j])
            # print(row[first + j])
    labels = np.empty(3)
    for i in range(3):
        row = table.row_values(8 + i)
        labels[i] = row[first]
    return data, labels


def loadTestData(n):
    totalData = np.empty((n, 12))
    totallabel = np.empty(n)
    data = xlrd.open_workbook('附录A.xlsx')
    table = data.sheet_by_name('test')
    for i in range(n):
        first = 10 + 4 * i
        data, label = readOneData(table, first)
        totalData[i] = data
        totallabel[i] = label
    return totalData, totallabel


def readOneData(table, first):
    data = []
    for i in range(4):
        row = table.row_values(4 + i)
        for j in range(3):
            data.append(row[first + j])
    row = table.row_values(8)
    label = row[first + 3]
    return data, label


