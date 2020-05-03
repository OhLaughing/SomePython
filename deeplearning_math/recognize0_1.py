# 《深度学习的数学》 中识别0和1的例子，64个训练数据
import math

import numpy as np

import deeplearning_math.LoadData as LoadData

datas, labels = LoadData.loadTrainData()

w2 = np.random.random((3, 12))
b2 = np.random.random(3)

w3 = np.random.random((2, 3))
b3 = np.random.random(2)

C = np.empty(64)

rate = 0.1
iteration_time = 500


def sigmoid(i):
    return 1 / (1 + math.exp(-1 * i))


def derivative_sigmoid(i):
    return sigmoid(i) * (1 - sigmoid(i))


# 预估参数
for iter in range(iteration_time):
    z2 = np.empty(3)
    a2 = np.empty(3)
    derivative_z2 = np.empty(3)
    z3 = np.empty(2)
    a3 = np.empty(2)
    derivative_z3 = np.empty(2)

    # 神经单元误差
    delta3 = np.empty(2)
    delta2 = np.empty(3)

    w3_delta = np.zeros((2, 3))
    b3_delta = np.zeros(2)
    w2_delta = np.zeros((3, 12))
    b2_delta = np.zeros(3)
    for i in range(64):
        a1 = datas[i]
        label = labels[i]
        for j in range(3):
            z2[j] = a1.dot(w2[j]) + b2[j]
            a2[j] = sigmoid(z2[j])
            derivative_z2[j] = derivative_sigmoid(z2[j])
        for j in range(2):
            z3[j] = a2.dot(w3[j]) + b3[j]
            a3[j] = sigmoid(z3[j])
            derivative_z3[j] = derivative_sigmoid(z3[j])
        t0 = 1 if label == 0 else 0
        t1 = 1 if label == 1 else 0
        # P149 (7)(8)
        delta3 = (a3 - np.array([t0, t1])) * derivative_z3
        delta3_t = np.tile([[delta3[0]], [delta3[1]]], (1, 3))
        w3_delta += (delta3_t * a2)
        b3_delta += delta3

        # P151 (15)
        delta2 = delta3.dot(w3) * derivative_z2
        delta2_t = np.tile([[delta2[0]], [delta2[1]], [delta2[2]]], (1, 12))
        w2_delta += (delta2_t * a1)
        b2_delta += delta2

        # w3_delta +=
        C[i] = 0.5 * (math.pow(t0 - a3[0], 2) + math.pow(t1 - a3[1], 2))
    w2 -= (rate * w2_delta)
    w3 -= (rate * w3_delta)
    b2 -= (rate * b2_delta)
    b3 -= (rate * b3_delta)
    c = np.sum(C)
    print('iteration time: ' + str(iter) + ', Ct=' + str(c))

print('w2')
print(w2)
print('b2')
print(b2)
print('w3')
print(w3)
print('b3')
print(b3)

right = 0

testnum = 10
datas, labels = LoadData.loadTestData(testnum)
# 测试
for i in range(testnum):
    a1 = datas[i]
    label = labels[i]
    z2 = np.empty(3)
    a2 = np.empty(3)
    z3 = np.empty(2)
    a3 = np.empty(2)
    for j in range(3):
        z2[j] = a1.dot(w2[j]) + b2[j]
        a2[j] = sigmoid(z2[j])
    for j in range(2):
        z3[j] = a2.dot(w3[j]) + b3[j]
        a3[j] = sigmoid(z3[j])

    if (a3[0] > a3[1] and label == 0) or (a3[0] < a3[1] and label == 1):
        right += 1
    else:
        print('the %d is wrong' % i)

print("right number: %d/%d " % (right, testnum))
