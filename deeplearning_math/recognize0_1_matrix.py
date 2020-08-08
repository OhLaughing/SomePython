# 《深度学习的数学》 中识别0和1的例子，64个训练数据

import numpy as np

import deeplearning_math.LoadData as LoadData

datas, labels = LoadData.loadTrainData()

a1 = datas

w2 = np.random.random((3, 12))
b2 = np.random.random(3)

w3 = np.random.random((2, 3))
b3 = np.random.random(2)


rate = 0.1
iteration_time = 500


def sigmoid(i):
    return 1 / (1 + np.exp(-1 * i))

def derivative_sigmoid(i):
    return sigmoid(i) * (1 - sigmoid(i))


# 预估参数
for iter in range(iteration_time):

    z2 = np.dot(datas, w2.transpose()) + b2.transpose()
    a2 = sigmoid(z2)
    z2_derivative = derivative_sigmoid(z2)

    z3 = np.dot(a2, w3.transpose()) + b3.transpose()
    a3 = sigmoid(z3)
    z3_derivative = derivative_sigmoid(z3)

    t = np.zeros((len(labels), 2))
    for i in range(len(labels)):
        t[i, int(labels[i])] = 1

    Ct = np.sum(np.power(a3 - t, 2)) / 2

    # 64*2
    delta3 = (a3 - t) * z3_derivative

    # 64*3
    delta2 = np.dot(delta3, w3) * z2_derivative

    w3_delta = np.zeros((len(labels), w3.shape[0], w3.shape[1]))

    for i in range(delta3.shape[0]):
        t = np.dot(delta3[i, :].reshape(2, 1), a2[i, :].reshape(1, 3))
        w3_delta[i, :, :] = t

    # w3_delta = np.dot(delta3.transpose(), a2)
    b3_delta = delta3.copy()

    # delta2 =

    w2_delta = np.empty((64, 3, 12))
    for i in range(64):
        w2_delta[i, :, :] = np.dot(delta2[i, :].reshape(3, 1), a1[i, :].reshape(1, 12))

    b2_delta = delta2.copy()

    w2_delta = np.sum(w2_delta, axis=0)
    w3_delta = np.sum(w3_delta, axis=0)
    b2_delta = np.sum(b2_delta, axis=0)
    b3_delta = np.sum(b3_delta, axis=0)

    w2 -= (rate * w2_delta)
    w3 -= (rate * w3_delta)
    b2 -= (rate * b2_delta)
    b3 -= (rate * b3_delta)
    print('iteration time: ' + str(iter) + ', Ct=' + str(Ct))

right = 0

testnum = 10

def check(datas, labels):
    z2 = np.dot(datas, w2.transpose()) + b2.transpose()
    a2 = sigmoid(z2)

    z3 = np.dot(a2, w3.transpose()) + b3.transpose()
    a3 = sigmoid(z3)
    out = a3.argmax(axis=1)
    same = out == labels
    print('check rate: '+ str(100* np.sum(same) / len(same)) + "%")

for i in range(len(labels)):
    labels[i] = int(labels[i])
check(datas, labels)

