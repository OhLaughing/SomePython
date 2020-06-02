import matplotlib.pyplot as plt
import numpy as np


def loadData_2feature(file):
    f = open(file, 'r')
    lines = f.readlines()
    x = []
    y = []
    for line in lines:
        line = line.strip('\n')
        v = line.split(',')
        x.append(float(v[0]))
        x.append(float(v[1]))
        y.append(float(v[2]))
    x = np.array(x)
    x = x.reshape(len(y), 2)
    y = np.array(y)
    y = y.reshape(len(y), 1)
    return x, y


def plot(x, y):
    exam1_x = []
    exam1_y = []
    exam2_x = []
    exam2_y = []
    for i in range(len(y)):
        if (y[i] == 0):
            exam1_x.append(x[i, 0])
            exam1_y.append([x[i, 1]])
        elif (y[i] == 1):
            exam2_x.append(x[i, 0])
            exam2_y.append([x[i, 1]])
    plt.scatter(exam1_x, exam1_y, s=20, c='r', label='not pass')
    plt.scatter(exam2_x, exam2_y, s=20, c='b', label='pass')
    plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


if __name__ == '__main__':
    x, y = loadData_2feature('ex2data1.txt')
    print(x)
    print(x.shape)
    print(y)
    print(y.shape)
    # plot(x, y)
    print(sigmoid(y))
