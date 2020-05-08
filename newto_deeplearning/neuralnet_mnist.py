# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


'''
《深度学习入门》 第三章
检测数字的概率，把28*28的图向左上移动一个格，看看检测的成功率是否有影响
'''



def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

def move_left_up(input):
    b = np.pad(input, pad_width=1, mode='constant', constant_values=0)
    return b[2:31,2:31].reshape(784)

x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    input = np.array(x[i]).reshape(28,28)
    input = move_left_up(input)
    y = predict(network, input)
    p= np.argmax(y) # 获取概率最高的元素的索引
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))