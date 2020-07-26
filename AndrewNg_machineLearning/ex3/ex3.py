from scipy.io import loadmat
from AndrewNg_machineLearning.ex2 import ex2
import numpy as np
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


if __name__ == '__main__':
    data = loadmat('ex3data1.mat')

    X = data['X']
    y = data['y']
    print(X.shape)
    print(y.shape)
    print(X[0])
    print(y[0:10])
