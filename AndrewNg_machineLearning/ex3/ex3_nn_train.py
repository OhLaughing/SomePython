from scipy.io import loadmat

from AndrewNg_machineLearning.Utils import *

# 本例子自己根据数据训练神经网络, 但还没有能正确的识别能力，有待优化

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


def derivative_sigmoid(i):
    return sigmoid(i) * (1 - sigmoid(i))


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
    # 有的隐藏层激活函数用relu有的用sigmoid，本实验的参数是基于sigmoid得出的
    a2 = sigmoid(z2)
    a2 = np.c_[np.ones(len(a2)), a2]
    z3 = np.dot(a2, theta2)
    a3 = sigmoid(z3)
    out = a3.argmax(axis=1)
    out = out + 1
    # out[out == 0] = 10
    out = out.reshape(len(out), 1)
    same = out == y
    print(np.sum(same) / len(same))


def train(datas, labels, iteration_time, rate, w2, b2, w3, b3):
    a1 = datas
    # 预估参数
    for iter in range(iteration_time):

        z2 = np.dot(datas, w2.transpose()) + b2.transpose()
        a2 = sigmoid(z2)
        z2_derivative = derivative_sigmoid(z2)

        z3 = np.dot(a2, w3.transpose()) + b3.transpose()
        a3 = sigmoid(z3)
        z3_derivative = derivative_sigmoid(z3)

        t = np.zeros((len(labels), 10))
        for i in range(len(labels)):
            index = int(labels[i])
            if (index == 10):
                index = 0
            t[i, index] = 1

        # 64*2
        delta3 = (a3 - t) * z3_derivative

        # 64*3
        delta2 = np.dot(delta3, w3) * z2_derivative

        w3_delta = np.zeros((len(labels), w3.shape[0], w3.shape[1]))

        for i in range(delta3.shape[0]):
            t = np.dot(delta3[i, :].reshape(10, 1), a2[i, :].reshape(1, 25))
            w3_delta[i, :, :] = t

        # w3_delta = np.dot(delta3.transpose(), a2)
        b3_delta = delta3.copy()

        # delta2 =

        w2_delta = np.empty((5000, 25, 400))
        for i in range(5000):
            w2_delta[i, :, :] = np.dot(delta2[i, :].reshape(25, 1), a1[i, :].reshape(1, 400))

        b2_delta = delta2.copy()

        w2_delta = np.sum(w2_delta, axis=0)
        w3_delta = np.sum(w3_delta, axis=0)
        b2_delta = np.sum(b2_delta, axis=0)
        b3_delta = np.sum(b3_delta, axis=0)

        w2 -= (rate * w2_delta)
        w3 -= (rate * w3_delta)
        b2 -= (rate * b2_delta)
        b3 -= (rate * b3_delta)

        Ct = np.sum(np.power(a3 - t, 2)) / 2
        print('iteration time: ' + str(iter) + ', Ct=' + str(Ct))
    return w2, b2, w3, b3


def checkRate(datas, labels, w2, b2, w3, b3):
    z2 = np.dot(datas, w2.transpose()) + b2.transpose()
    a2 = sigmoid(z2)

    z3 = np.dot(a2, w3.transpose()) + b3.transpose()
    a3 = sigmoid(z3)
    out = a3.argmax(axis=1)
    out[out == 0] = 10
    out = out.reshape(len(out), 1)
    same = out == labels
    print(np.sum(same) / len(same))
    print(np.sum(same))
    print(len(same))


if __name__ == '__main__':
    # 
    data = loadmat('ex3data1.mat')

    X = data['X']
    y = data['y']

    w2 = np.loadtxt('w2.txt')
    w3 = np.loadtxt('w3.txt')
    b2 = np.loadtxt('b2.txt')
    b3 = np.loadtxt('b3.txt')

    w2, b2, w3, b3 = train(X, y, 100, 0.05, w2, b2, w3, b3)
    # np.savetxt('w2.txt', w2)
    # np.savetxt('b2.txt', b2)
    # np.savetxt('w3.txt', w3)
    # np.savetxt('b3.txt', b3)

    checkRate(X, y, w2, b2, w3, b3)
