import numpy as np
from scipy.io import loadmat


def sigmoid(z):
    '''
    z为prev_activation, size为 nl * m
    '''
    return 1 / (1 + np.exp(-z))


def relu(z):
    '''
    z为prev_activation, size为 nl * m
    '''
    return np.maximum(0, z)


def cost_nn(x, y, theta1, theta2):
    X = np.c_[np.ones(len(x)), x]
    z2 = np.dot(X, theta1)
    a2 = relu(z2)
    a2 = np.c_[np.ones(len(a2)), a2]
    z3 = np.dot(a2, theta2)
    a3 = sigmoid(z3)
    out = np.zeros((len(y), 10))
    for i in range(len(y)):
        index = 0 if y[i] == 10 else y[i]
        out[i, index] = 1
    a3 = a3 - out
    return np.sum(np.power(a3, 2))


def check(x, y, theta1, theta2):
    X = np.c_[np.ones(len(x)), x]
    z2 = np.dot(X, theta1)
    a2 = relu(z2)
    a2 = np.c_[np.ones(len(a2)), a2]
    z3 = np.dot(a2, theta2)
    a3 = sigmoid(z3)
    out = a3.argmax(axis=1)
    out = out + 1

    out[out == 0] = 10
    out = out.reshape(len(out), 1)
    same = out == y
    print(np.sum(same) / len(same))


if __name__ == '__main__':
    data = loadmat('ex3data1.mat')
    weights = loadmat('ex3weights.mat')
    # 隐藏层神经单元个数
    layer2_unit = 25
    # 输出神经单元个数
    out_unit = 10

    X = data['X']
    y = data['y']

    print(X.shape)
    print(y.shape)
    theta1 = np.ones((X.shape[1] + 1, layer2_unit))
    theta2 = np.ones((layer2_unit + 1, out_unit))
    theta1 = weights['Theta1']
    theta1 = theta1.transpose()
    theta2 = weights['Theta2']
    theta2 = theta2.transpose()
    cost = cost_nn(X, y, theta1, theta2)
    check(X, y, theta1, theta2)
