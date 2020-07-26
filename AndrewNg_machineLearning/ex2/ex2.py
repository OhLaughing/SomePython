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
    left = -1 * np.log(sigmoid(X.dot(theta)) + epsilon).T.dot(y)
    right = (y - 1).T.dot(np.log(1 - sigmoid(X.dot(theta))) + epsilon)
    print('left: %f, right: %f' % (left, right))
    return (left + right) / len(y)


def costFunction_reg(theta, x, y, rate):
    epsilon = 1e-5
    X = np.c_[np.ones(len(x)), x]
    left = -1 * np.log(sigmoid(X.dot(theta)) + epsilon).T.dot(y)
    right = (y - 1).T.dot(np.log(1 - sigmoid(X.dot(theta))) + epsilon)
    print(theta.shape)
    print(theta.T.dot(theta))
    reg = (rate / 2) * (theta.T.dot(theta) - theta[0] * theta[0])
    print('type(reg):' + str(type(reg)))
    return (left + right + reg) / len(y)


# 正则化的梯度计算,只计算步长
def gridentDescent_reg(theta, x, y, rate):
    X = np.c_[np.ones(len(x)), x]
    a = sigmoid(X.dot(theta))
    a = a.reshape(len(x), 1)
    reg = (rate / len(x)) * theta
    print('regshape' + str(reg.shape))
    reg[0] = 0
    tmp = X.T.dot(a - y)/len(x)
    print(tmp.shape)
    dtheta = np.empty(len(tmp))
    for i in range(len(tmp)):
        dtheta[i] = tmp[i]+reg[i]
    print(dtheta.shape)
    return dtheta


def gridentDescent(theta, x, y, alpha, iters):
    X = np.c_[np.ones(len(x)), x]
    for i in range(iters):
        print('C ' + str(costFunction(theta, x, y)))
        dtheta = X.T.dot(sigmoid(X.dot(theta)) - y) / len(x)
        theta -= dtheta * alpha
        # print(theta)
    return theta


# 该方法作为opt.fmin_tnc方法的入参，该方法并没有通过迭代梯度下降，而是只计算了步长
def gridentDescent_1(theta, x, y):
    X = np.c_[np.ones(len(x)), x]
    a = sigmoid(X.dot(theta))
    a = a.reshape(len(x), 1)
    dtheta = X.T.dot(a - y) / len(x)
    return dtheta


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
    theta = np.zeros(3).reshape(3, 1)
    # x经过特征缩放与没有缩放的costfunction的值是相同的
    print(costFunction(theta, x, y))
    aver_range = ex1.getAverAndRange(x)
    x1 = ex1.featureScalling(x, aver_range)
    theta = gridentDescent(theta, x1, y, 0.1, 2000)
    print(theta)
    print('reverse featureScalling')
    theta = ex1.calculateTheta(theta, aver_range)
    print(theta)
    # plot(x,y)
    plotresult(x, y, theta)


# 使用fmin_tnc找最优值
def test1_fmin_tnc():
    x, y = loadData_2feature('ex2data1.txt')
    X = np.c_[np.ones(len(x)), x]
    # plot(x, y)
    # 为什么theta初始值全为1，就不行？
    theta = np.zeros(3).reshape(3, 1)

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


def test2_fmin_tnc():
    x, y = loadData_2feature('ex2data2.txt')

    l = len(y)
    X = np.empty(shape=(l, 27))

    c = 0
    for a in range(1, 7):
        for b in range(0, a + 1):
            newColumn = (x[:, 0] ** (a - b)) * (x[:, 1] ** b)

            X[:, c] = newColumn
            c = c + 1

    theta = np.zeros(28).reshape(28, 1)

    result = opt.fmin_tnc(func=costFunction, x0=theta, fprime=gridentDescent_1, args=(X, y))

    print(result)
    print(result[0])
    print(type(result))
    print("*" * 30)
    theta = result[0]

    x1 = np.arange(-0.75, 1, 0.01)
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
            if np.abs(Z[i, j]) < 0.2:
                p_y.append(x1[i])
                p_x.append(x2[j])

    plotresult1(x, y, p_x, p_y)


# 正则化
def test2_regularized(rate, error):
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

    theta = np.ones(28).reshape(28, 1)
    print('theta.shape'+str(theta.shape))
    result = opt.fmin_tnc(func=costFunction_reg, x0=theta, fprime=gridentDescent_reg, args=(X, y, rate))
    print(result)
    print(result[0])
    print(type(result))
    print("*" * 30)
    theta = result[0]

    x1 = np.arange(-0.75, 1.3, 0.01)
    x2 = np.arange(-0.75, 1.3, 0.01)

    X, Y = np.meshgrid(x1, x2)

    Z = theta[0]
    c = 1
    for a in range(1, 7):
        for b in range(0, a + 1):
            Z = Z + (X ** (a - b)) * (Y ** b) * theta[c]

            c = c + 1

    p_x = []
    p_y = []
    for i in range(205):
        for j in range(205):
            if np.abs(Z[i, j]) < error:
                p_y.append(x1[i])
                p_x.append(x2[j])

    plotresult1(x, y, p_x, p_y)

if __name__ == '__main__':
    # test1()
    # test2_regularized(0, 0.05)
    test2_regularized(1, 0.01)
    # test2_regularized(100, 0.0001)
    # test2_fmin_tnc()
