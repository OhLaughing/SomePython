import math

import numpy as np

import deeplearning_math.LoadData as LoadData


def sigmoid(i):
    if (i < -100):
        return 0
    else:
        return 1 / (1 + math.exp(-1 * i))


def derivative_sigmoid(i):
    return sigmoid(i) * (1 - sigmoid(i))


def relu(x):
    return np.maximum(0, x)


datas, labels = LoadData.loadCNNTrainData(96)

C = np.zeros(96)

# 卷积层
filters = np.random.random((3, 3, 3))

bF = np.random.random(3)

# 池化层
wOut = np.random.random((3, 3, 2, 2))

# 输出层
bOut = np.random.random(3)

iteration = 50
rate = 0.01
for loop in range(iteration):
    # 卷积层
    filter_delta = np.zeros((3, 3, 3))

    bF_delta = np.zeros(3)

    # 池化层
    delta_W_Out = np.zeros((3, 3, 2, 2))

    # 输出层
    delta_b_out = np.zeros(3)

    for i in range(96):
        # 数据
        data = datas[i]
        label = labels[i]

        # 卷积层
        zF = np.empty((3, 4, 4))
        aF = np.empty((3, 4, 4))
        delta_zF = np.empty((3, 4, 4))
        # 池化层
        zP = np.empty((3, 2, 2))
        aP = np.empty((3, 2, 2))

        # 输出层
        zOut = np.empty(3)
        aOut = np.empty(3)

        # 从输入层到卷积层、池化层
        for j in range(3):
            F = filters[j]
            for k in range(4):
                for m in range(4):
                    zF[j, k, m] = np.sum(F * data[k:k + 3, m:m + 3] + bF[j])
                    aF[j, k, m] = sigmoid(zF[j, k, m])

            zP[j, 0, 0] = np.max(zF[0:2, 0:2])
            zP[j, 0, 1] = np.max(zF[0:2, 2:4])
            zP[j, 1, 0] = np.max(zF[2:4, 0:2])
            zP[j, 1, 1] = np.max(zF[2:4, 2:4])
            aP[j] = relu(zP[j])

        # 从池化层到输出层
        for j in range(3):
            zOut[j] = np.sum(zP * wOut[j]) + bOut[j]
            aOut[j] = sigmoid(zOut[j])
        t1 = 1 if label[0] == 1 else 0
        t2 = 1 if label[1] == 1 else 0
        t3 = 1 if label[2] == 1 else 0

        # 平方误差
        C[i] = 0.5 * (math.pow(t1 - aOut[0], 2) + math.pow(t2 - aOut[1], 2) + math.pow(t3 - aOut[2], 2))

        # 输出层 神经单元误差 1*3
        delta_Out = (aOut - np.array([t1, t2, t3])) * [derivative_sigmoid(i) for i in zOut]
        delta_Out = np.array(delta_Out)
        # 池化层到输出层的权重误差
        delta_W_Out_Tmp = np.empty((3, 3, 2, 2))
        for j in range(3):
            delta_W_Out_Tmp[j] = aP * delta_Out[j]
        delta_W_Out += delta_W_Out_Tmp
        delta_b_out += delta_Out

        #     计算delta_zP 3*2*2 池化层
        delta_zP = np.empty((3, 2, 2))
        for j in range(3):
            for k in range(2):
                for l in range(2):
                    delta_zP_t = 0
                    for m in range(3):
                        delta_zP_t += wOut[m, j, k, l] * delta_Out[m]
                    delta_zP[j, k, l] = delta_zP_t
        # 计算delta_zF 3*4*4
        for j in range(3):
            for k in range(4):
                for l in range(4):
                    delta_zF[j, k, l] = delta_zP[j, int(k / 2), int(l / 2)] * derivative_sigmoid(zF[j, k, l]) \
                        if aF[j, k, l] == zP[j, int(k / 2), int(l / 2)] else 0

        # 卷积层 b 的误差
        for j in range(3):
            bF_delta[j] += np.sum(delta_zF)
            for k in range(3):
                for l in range(3):
                    filter_delta[j, k, l] += np.sum(data[k:k + 4, l:l + 4] * zF[k])

    wOut -= (delta_W_Out * rate)
    bOut -= (delta_b_out * rate)

    filters -= (filter_delta * rate)
    bF -= (bF_delta * rate)

    print('iteration time: ' + str(loop) + " Ct:" + str(np.sum(C)))
