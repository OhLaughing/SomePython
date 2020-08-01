import numpy as np
import scipy.optimize as opt
from scipy.io import loadmat

from AndrewNg_machineLearning.ex2 import ex2

# https://zhuanlan.zhihu.com/p/51355706
def gradient_with_loop(theta, x, y, rate):
    # 正则化的梯度计算,只计算步长
    X = np.c_[np.ones(len(x)), x]
    a = ex2.sigmoid(X.dot(theta))
    a = a.reshape(len(x), 1)
    reg = (rate / len(x)) * theta
    print('regshape' + str(reg.shape))
    reg[0] = 0
    tmp = X.T.dot(a - y) / len(x)
    print(tmp.shape)
    dtheta = np.empty(len(tmp))
    for i in range(len(tmp)):
        dtheta[i] = tmp[i] + reg[i]
    print(dtheta.shape)
    return dtheta

def check(X, y, theta):
    X = np.c_[np.ones(len(X)), X]
    t = np.dot(X,theta)
    t = t.argmax(axis=1)
    t[t == 0] = 10
    print(t.shape)
    t = t.reshape(5000,1)
    print(y.shape)
    same = t==y
    print(same.shape)
    print(np.sum(same)/len(same))

if __name__ == '__main__':
    data = loadmat('ex3data1.mat')

    X = data['X']
    y = data['y']
    print(y.shape)
    a=y.transpose()
    print(a.shape)
    print('X.shape' + str(X.shape))
    print('y.shape' + str(y.shape))
    classK = 10
    theta = np.zeros((X.shape[1] + 1, classK))
    # for i in range(classK):
    #     tmp = 10 if i == 0 else i
    #     Y = y == tmp
    #     Y = Y + 0
    #     print(Y)
    #     theTheta = np.zeros((X.shape[1] + 1))
    #     print((theTheta.shape))
    #     result = opt.fmin_tnc(func=ex2.costFunction, x0=theTheta, fprime=ex2.gridentDescent_1, args=(X, Y))
    #     theTheta = result[0]
    #     theta[:,i] = theTheta
    #     print(theTheta.shape)
    # np.savetxt('file1.txt', theta)
    theta = np.loadtxt('file1.txt')
    print(theta)
    print(theta.shape)
    check(X, y, theta)
        check(X, y, theta)
    newX = np.empty_like(X)
    for i in range(len(X)):
        line = X[i]
        newX[i] = move(line, 20, 20, -2, -2)
    check(newX, y, theta)
