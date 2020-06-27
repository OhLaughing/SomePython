import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

# import sysloadData_2feature
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


def costFunction(theta, x, y):
    epsilon = 1e-5
    X = np.c_[np.ones(len(x)), x]
    return (-1 * np.log(sigmoid(X.dot(theta))).T.dot(y) + (y - 1).T.dot(np.log(1 - sigmoid(X.dot(theta))))) / len(y)


def costFunction_reg(x, y, theta, rate):
    epsilon = 1e-5
    X = np.c_[np.ones(len(x)), x]
    return (-1 * np.log(sigmoid(X.dot(theta))).T.dot(y) + (y - 1).T.dot(np.log(1 - sigmoid(X.dot(theta))))) / len(y) + (
            rate / 2 * len(y)) * theta.T.dot(theta)


def gridentDescent(theta, x, y, alpha, iters):
    X = np.c_[np.ones(len(x)), x]
    for i in range(iters):
        print('C ' + str(costFunction(theta, x, y)))
        dtheta = X.T.dot(sigmoid(X.dot(theta)) - y) / len(x)
        theta -= dtheta * alpha
        # print(theta)
    return theta

def gridentDescent_1(theta, x, y):
    X = np.c_[np.ones(len(x)), x]
    for i in range(1000):
        print('C ' + str(costFunction(theta, x, y)))
        dtheta = X.T.dot(sigmoid(X.dot(theta)) - y) / len(x)
        theta -= dtheta * 0.1
        # print(theta)
    return theta


def gridentDescent_reg(x, y, theta, alpha, iters):
    X = np.c_[np.ones(len(x)), x]
    for i in range(iters):
        print('C ' + str(costFunction_reg(x, theta, y)))
        dtheta = X.T.dot(sigmoid(X.dot(theta)) - y) / len(x)
        theta -= dtheta * alpha
        print(theta)
    return theta


def plotresult(x, y, theta):
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
    plt.scatter(exam1_x, exam1_y, s=20, c='r', marker='o', label='not pass')
    plt.scatter(exam2_x, exam2_y, s=20, c='b', marker="x", label='pass')
    x = np.linspace(30, 100, 10)
    y = (-1 * theta[0] - theta[1] * x) / theta[2]
    plt.plot(x, y, '-r')
    plt.show()


def plotresult1(x, y, p_x, p_y):
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
    plt.scatter(exam1_x, exam1_y, s=20, c='r', marker='o', label='not pass')
    plt.scatter(exam2_x, exam2_y, s=20, c='b', marker="x", label='pass')
    plt.scatter(p_x, p_y, s=20, c='y', marker="+", label='pass')

    plt.show()


# 使用梯度下降法找最优值
def test1():
    x, y = loadData_2feature('ex2data1.txt')

    # plot(x, y)
    theta = np.ones(3).reshape(3, 1)
    aver_range = ex1.getAverAndRange(x)
    x1 = ex1.featureScalling(x, aver_range)
    theta = gridentDescent(theta, x1, y, 0.1, 2000)
    print(theta)
    print('reverse eatureScalling')
    theta = ex1.calculateTheta(theta, aver_range)
    print(theta)
    # plot(x,y)
    plotresult(x, y, theta)


# 使用fmin_tnc找最优值
def test1_fmin_tnc():
    x, y = loadData_2feature('ex2data1.txt')
    X = np.c_[np.ones(len(x)), x]
    # plot(x, y)
    theta = np.ones(3).reshape(3, 1)

    result = opt.fmin_tnc(func=costFunction, x0=theta, fprime=gridentDescent_1, args=(x, y))
    print(result)


def test2():
    x, y = loadData_2feature('ex2data2.txt')

    l = len(y)
    X = np.empty(shape=(l, 27))
    print(X.shape)

    c = 0
    for a in range(1, 7):
        for b in range(0, a + 1):
            newColumn = (x[:, 0] ** (a - b)) * (x[:, 1] ** b)

            X[:, c] = newColumn
            c = c + 1

    theta = np.zeros(28).reshape(28, 1)
    aver_range = ex1.getAverAndRange(X)
    X1 = ex1.featureScalling(X, aver_range)
    theta = gridentDescent(theta, X1, y, 0.1, 1000)
    theta = ex1.calculateTheta(theta, aver_range)
    print("*" * 30)
    print(theta)
    print(theta.shape)

    x1 = np.arange(-0.75, 1, 0.01)
    print(len(x1))
    x2 = np.arange(-0.75, 1, 0.01)

    X, Y = np.meshgrid(x1, x2)

    Z = theta[0]
    c = 1
    for a in range(1, 7):
        for b in range(0, a + 1):
            Z = Z + (X ** (a - b)) * (Y ** b) * theta[c]

            c = c + 1

    p_x = []
    p_y = []
    for i in range(175):
        for j in range(175):
            if np.abs(Z[i, j]) < 1e-2:
                p_y.append(x1[i])
                p_x.append(x2[j])

    plotresult1(x, y, p_x, p_y)


# 正则化
def test2_reg():
    x, y = loadData_2feature('ex2data2.txt')

    l = len(y)
    X = np.empty(shape=(l, 27))
    print(X.shape)

    c = 0
    for a in range(1, 7):
        for b in range(0, a + 1):
            newColumn = (x[:, 0] ** (a - b)) * (x[:, 1] ** b)

            X[:, c] = newColumn
            c = c + 1

    theta = np.zeros(28).reshape(28, 1)
    aver_range = ex1.getAverAndRange(X)
    X1 = ex1.featureScalling(X, aver_range)
    theta = gridentDescent(theta, X1, y, 0.1, 3000)
    theta = ex1.calculateTheta(theta, aver_range)
    print("*" * 30)
    print(theta)
    print(theta.shape)

    x1 = np.arange(-0.75, 1, 0.01)
    print(len(x1))
    x2 = np.arange(-0.75, 1, 0.01)

    X, Y = np.meshgrid(x1, x2)

    Z = theta[0]
    c = 1
    for a in range(1, 7):
        for b in range(0, a + 1):
            Z = Z + (X ** (a - b)) * (Y ** b) * theta[c]

            c = c + 1

    p_x = []
    p_y = []
    for i in range(175):
        for j in range(175):
            if np.abs(Z[i, j]) < 1e-2:
                p_y.append(x1[i])
                p_x.append(x2[j])

    plotresult1(x, y, p_x, p_y)


if __name__ == '__main__':
    # test1()
    test1_fmin_tnc()
