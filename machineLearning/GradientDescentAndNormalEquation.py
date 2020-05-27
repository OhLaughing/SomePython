import numpy as np

import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
# 梯度下降法和正规方程法比较

# 本例是一个特征值的情况， 特征为房屋面积，目标为房屋价格

# 首先构造数据
# 房屋面积x从30-300，随机取30个值
x = np.random.randint(30, 300, 30).reshape(30, 1)

# y = 2*x + 10 上下浮动

y = (2 * x) + 10 + (np.random.randn(30).reshape(30, 1) * 30)


# plt.scatter(x, y)
# plt.show()


def gradientDescent(x, y):
    rate = 0.0000003
    iterator = 1000
    theta = np.random.rand(2).reshape(2, 1)
    X = np.hstack((np.ones(30).reshape(30, 1), x))

    for i in range(iterator):
        t = X.dot(theta) - y
        C = (X.dot(theta) - y).T.dot(X.dot(theta) - y)
        print('iterator time: ' + str(i) + ' C： ' + str(C))
        theta_delta = np.array([np.sum(t), np.sum(t * x)]).reshape(2, 1) / 30
        d_theta = X.T.dot(t) / 30
        theta -= (theta_delta * rate)
        print('theta:  \n' + str(theta))
        # print('theta_delta:  ' + str(theta_delta))
    return theta


def graddientDescent_other(X, Y):
    X = np.hstack((np.ones(30).reshape(30, 1), X))
    w = np.random.random(size=(2, 1))
    # 更新参数
    epoches = 1000
    eta = 0.0000003
    losses = []  # 记录loss变化
    for _ in range(epoches):
        dw = -2 * X.T.dot(Y - X.dot(w))
        w = w - eta * dw
        losses.append((Y - X.dot(w)).T.dot(Y - X.dot(w)).reshape(-1))
        # print(w)
    return w,losses


def gradientDescent_featureScaling(x, y):
    x = (x - 150) / 300
    return gradientDescent(x, y)


# gradientDescent(x, y)
def normalEquation(x, y):
    X = np.hstack((np.ones(30).reshape(30, 1), x))
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), y)
    print('normalEquation theta: \n%s' % theta)
    print('cost function:' + str((y-X.dot(theta)).T.dot(y-X.dot(theta))))
    n = np.arange(0, 300, 10)
    m = theta[0] + n * theta[1]
    plt.scatter(x, y)
    plt.plot(n, m, 'r')
    # plt.show()


if __name__ == '__main__':
    print('hh')

    feature_mean = np.mean(x, axis=0)
    feature_std = np.std(x, axis=0) + 1e-8
    x = (x - feature_mean) / feature_std
    theta = gradientDescent(x, y)
    print(theta)
    # w ,loss= graddientDescent_other(x, y)
    # print(x)
    # print('grident descent of other person: ')
    # print(w)
    # plt.plot(loss)
    normalEquation(x,y)
    X = np.hstack((np.ones(30).reshape(30, 1), x))
    theta=np.array([10,2]).reshape(2,1)
    print('cost function [10,2]:'+str((y-X.dot(theta)).T.dot(y-X.dot(theta))))

    plt.scatter(x, y)
    plt.show()

    n = np.arange(0, 300, 10)
    m = 10 + n * 2
    plt.plot(n, m, 'b')

    # plt.show()
# 感觉梯度下降法还是有点问题，theta_0 偏差有点大，可能是需要特征缩放
