import matplotlib.pyplot as plt
import numpy as np

# import sys
# sys.path.append("..\\ex1")
from AndrewNg_machineLearning.ex1 import ex1


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


def costFunction(x, theta, y):
    epsilon = 1e-5
    X = np.c_[np.ones(len(x)), x]
    return (-1 * np.log(X.dot(theta) + epsilon).T.dot(y) + (y - 1).T.dot(np.log(1 - X.dot(theta) + epsilon))) / len(y)


def gridentDescent(x, y, theta, alpha, iters):
    X = np.c_[np.ones(len(x)), x]
    for i in range(iters):
        print('C ' + str(costFunction(x, theta, y)))
        dtheta = X.T.dot(sigmoid(X.dot(theta)) - y) / len(x)
        theta -= dtheta * alpha
        print(theta)
    return theta


if __name__ == '__main__':
    x, y = loadData_2feature('ex2data1.txt')

    # plot(x, y)
    theta = np.zeros(3).reshape(3, 1)
    aver_range = ex1.getAverAndRange(x)
    x1 = ex1.featureScalling(x, aver_range)
    theta = gridentDescent(x1, y, theta, 0.1, 2000)
    print(theta)
