import numpy as np
import xlrd


# 读取excel中的64个测试数据
def loadTrainData():
    totalData = np.empty((64, 12))
    totallabel = np.empty(64)
    data = xlrd.open_workbook('3-5 NN（求解器）.xlsx')
    table = data.sheet_by_name('Data')
    for i in range(64):
        first = 10 + 4 * i
        data, label = readOneData(table, first)
        totalData[i] = data
        totallabel[i] = label
    return totalData, totallabel

def loadTestData(n):
    totalData = np.empty((n, 12))
    totallabel = np.empty(n)
    data = xlrd.open_workbook('3-5 NN（求解器）.xlsx')
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

