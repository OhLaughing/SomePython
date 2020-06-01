import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def loadData(file):
    f = open(file, 'r')
    lines = f.readlines()
    x = []
    y = []
    for line in lines:
        line = line.strip('\n')
        v = line.split(',')
        x.append(float(v[0]))
        y.append(float(v[1]))
    return x, y


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


def plotData(x, y):
    plt.scatter(x, y, 15, 'r', 'x')
    plt.xlabel('Population of City in 10000s')
    plt.ylabel('Profit in $10000s')


def scale(c):
    return (c - np.average(c)) / (np.max(c) - np.min(c))


def featureScalling(x, aver_range):
    if x.ndim == 1:
        return (x - aver_range[0]) / aver_range[1]
    for i in range(x.shape[1]):
        c = x[:, i]
        c = (c - aver_range[0, i]) / aver_range[1, i]
        x[:, i] = c
    return x


def gradientDescent(x, y, theta, alpha, iters):
    X = np.c_[np.ones(len(x)), x]
    for i in range(iters):
        C = ((X.dot(theta) - y).T.dot(X.dot(theta) - y)) / (2 * len(x))
        print("cost function: " + str(C))
        print(theta)
        dtheta = X.T.dot(X.dot(theta) - y) / len(x)
        theta = theta - (dtheta * alpha)
    return theta


def normalEquation(x, y):
    X = np.c_[np.ones(len(x)), x]
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), y)
    return theta


def gradientDescentMulVariables(x, y, theta, alpha, iters):
    X = np.c_[np.ones(len(x)), x]

    for i in range(iters):
        a = X.dot(theta)
        print(a.shape)
        print(y.shape)
        error = a - y
        C = error.T.dot(error) / (2 * len(x))
        print("C: " + str(C))
        dtheta = X.T.dot(X.dot(theta) - y) / len(x)
        print(theta)
        theta = theta - (dtheta * alpha)
    return theta


def plotLine(w):
    x = np.linspace(0, 20, 10)
    y = x * w[1] + w[0]
    plt.plot(x, y, '-r')


def costFunction(x, theta, y):
    X = np.c_[np.ones(len(x)), x]
    error = X.dot(theta) - y
    C = error.T.dot(error) / (2 * len(x))
    return C


def plotcostFunction3D(x, y):
    fig1 = plt.figure()
    ax = Axes3D(fig1)
    w0, w1 = np.mgrid[-2:2:40j, -2:2:40j]
    J = .0
    for i in range(97):
        J += (w0 + w1 * x[i] - y[i]) ** 2

    plt.title("This is main title")  # 总标题
    ax.plot_surface(w0, w1, J, rstride=1, cstride=1, cmap=plt.cm.coolwarm, alpha=0.5)  # 用取样点(x,y,z)去构建曲面
    ax.set_xlabel('x label', color='r')
    ax.set_ylabel('y label', color='g')
    ax.set_zlabel('z label', color='b')  # 给三个坐标轴注明
    plt.show()  # 显示模块中的所有绘图对象


# 等高线，即代价函数相同
def plotContour(x, y):
    step = 20
    a = np.arange(-20000, 20000, step)
    b = np.arange(-5000, 5000, step)
    A, B = np.meshgrid(a, b)
    J = 0
    for i in range(97):
        J += (A + B * x[i] - y[i]) ** 2
    # 画等高线
    plt.contour(A, B, J)
    plt.show()


def testMultipleVariable():
    x, y = loadData_2feature('ex1data2.txt')
    theta = np.zeros(3).reshape(3, 1)
    aver_range = getAverAndRange(x)

    x = featureScalling(x, aver_range)
    theta = gradientDescentMulVariables(x, y, theta, 0.01, 1000)
    theta = calculateTheta(theta, aver_range)
    print('gradientDescentMulVariables' + str(theta))
    C = costFunction(x, theta, y)
    print('C: ' + str(C))
    w = normalEquation(x, y)
    print("normalEquation" + str(w))
    C = costFunction(x, w, y)
    print('C: ' + str(C))


def getAverAndRange(x):
    '''
    :param x:输入x的一列为一个特征的样本值
    :return:x有几列，输出就有几列，并且行数为2，第一行为平均值，第二行为范围
    '''
    if (x.ndim == 1):
        result = np.empty((2, 1))
        result[0] = np.average(x)
        result[1] = np.max(x) - np.min(x)
        return result
    elif (x.ndim == 2):
        column = x.shape[1]
        result = np.empty((2, column))
        for i in range(column):
            result[0, i] = np.average(x[:, i])
            result[1, i] = np.max(x[:, i]) - np.min(x[:, i])
        return result


def calculateTheta(theta, aver_range):
    for i in range(len(theta) - 1):
        theta[0] -= theta[i + 1] * aver_range[0, i] / aver_range[1, i]
    for i in range(len(theta) - 1):
        theta[i + 1] = theta[i + 1] / aver_range[1, i]
    return theta


def testOneVariable():
    x, y = loadData('ex1data1.txt')
    x = np.array(x)
    # plotcostFunction3D(x, y)
    # plotContour(x,y)
    print(x.shape)
    aver_range = getAverAndRange(x)
    x = featureScalling(x, aver_range)
    print(x)
    # print(x[0:4])
    theta = np.zeros(2)
    w = gradientDescent(x, y, theta, 0.1, 1000)
    print(w)

    w = calculateTheta(w, aver_range)
    print(w)
    C = costFunction(x, w, y)
    print("C: " + str(C))
    w = normalEquation(x, y)
    print('normal equation' + str(w))
    C = costFunction(x, w, y)
    print("C: " + str(C))
    # X = np.c_[np.ones(len), x]
    # C = (X.dot(w) - y).T.dot(X.dot(w) - y)/(2*97)
    # for i in range(100):
    #     C1 = (X.dot(w) - y).T.dot(X.dot(w) - y)/(2*97)
    #     print('C: ' + str(C1))
    #     if(C1 > C):
    #         break
    #     C = C1
    #     w[0]+=0.01

    # plotData(x, y)
    #
    # plotLine(w)
    # plt.show()


if __name__ == '__main__':
    # testOneVariable()

    testOneVariable()
