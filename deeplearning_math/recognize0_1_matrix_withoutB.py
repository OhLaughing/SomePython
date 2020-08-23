# 《深度学习的数学》 中识别0和1的例子，64个训练数据

import numpy as np

import deeplearning_math.LoadData as LoadData

datas, labels = LoadData.loadTrainData()
# 60*13
a1 = np.c_[np.ones(datas.shape[0]), datas]

w2 = np.random.random((3, 12 + 1))

w3 = np.random.random((2, 3 + 1))

rate = 0.1
iteration_time = 500


def sigmoid(i):
    return 1 / (1 + np.exp(-1 * i))


def derivative_sigmoid(i):
    return sigmoid(i) * (1 - sigmoid(i))


# 预估参数
for iter in range(iteration_time):
    # 60*3
    z2 = np.dot(a1, w2.transpose())
    # 60*4
    a2 = np.c_[np.ones(datas.shape[0]), sigmoid(z2)]
    # 60*3
    z2_derivative = derivative_sigmoid(z2)

    # 60*2
    z3 = np.dot(a2, w3.transpose())
    a3 = sigmoid(z3)
    z3_derivative = derivative_sigmoid(z3)

    t = np.zeros((datas.shape[0], 2))
    for i in range(datas.shape[0]):
        t[i, int(labels[i])] = 1

    Ct = np.sum(np.power(a3 - t, 2)) / 2

    # 64*2
    delta3 = (a3 - t) * z3_derivative

    # 64*3
    delta2 = np.dot(delta3, w3[:,:-1]) * z2_derivative

    # 60*2*4
    w3_delta = np.zeros((len(labels), w3.shape[0], w3.shape[1]))

    for i in range(delta3.shape[0]):
        tmp = np.dot(delta3[i, :].reshape(2, 1), a2[i, :].reshape(1, 4))
        w3_delta[i, :, :] = tmp


    w2_delta = np.empty((64, 3, 13))

    for i in range(64):
        w2_delta[i, :, :] = np.dot(delta2[i, :].reshape(3, 1), a1[i, :].reshape(1, 13))

    w2_delta = np.sum(w2_delta, axis=0)
    w3_delta = np.sum(w3_delta, axis=0)

    w2 -= (rate * w2_delta)
    w3 -= (rate * w3_delta)

    print('iteration time: ' + str(iter) + ', Ct=' + str(Ct))


def check(a1, labels):
    z2 = np.dot(a1, w2.transpose())
    # 60*4
    a2 = np.c_[np.ones(datas.shape[0]), sigmoid(z2)]

    z3 = np.dot(a2, w3.transpose())
    a3 = sigmoid(z3)
    out = a3.argmax(axis=1)
    same = out == labels
    print('check rate: ' + str(100 * np.sum(same) / len(same)) + "%")


for i in range(len(labels)):
    labels[i] = int(labels[i])
check(a1, labels)
