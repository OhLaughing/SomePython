import numpy as np

# 梯度下降法和正规方程法比较

# 本例是一个特征值的情况， 特征为房屋面积，目标为房屋价格

# 首先构造数据
# 房屋面积x从30-300，随机取30个值
x = np.random.randint(30, 300, 30).reshape(30, 1)

# y = 2*x + 10 上下浮动

y = (2 * x) + 10 + (np.random.randn(30).reshape(30, 1) * 40)

# plt.scatter(x, y)
# plt.show()

rate = 0.00001


def gradientDescent(x, y):
    iterator = 200
    theta = np.random.rand(2).reshape(2, 1)
    X = np.hstack((np.ones(30).reshape(30, 1), x))

    for i in range(iterator):
        t = np.dot(X, theta) - y
        C = np.sum(np.power(t, 2)) / (30 * 2)
        print('iterator time: ' + str(i) + ' C： ' + str(C))
        theta_delta = np.array([np.sum(t), np.sum(t * x)]).reshape(2, 1) / 30
        theta -= (theta_delta * rate)
        print('theta:  \n' + str(theta))
        # print('theta_delta:  ' + str(theta_delta))

    n = np.arange(0, 300, 10)
    m = theta[0] + n * theta[1]

    print('after gridient descent, theta_0: %f , theta_1: %f' % (theta[0], theta[1]))
    # plt.scatter(x, y)
    # plt.plot(n, m, 'r')
    # plt.show()


# gradientDescent(x, y)
def normalEquation(x, y):
    X = np.hstack((np.ones(30).reshape(30, 1), x))
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), y)
    print('normalEquation theta: \n%s' % theta)

    # n = np.arange(0, 300, 10)
    # m = theta[0] + n * theta[1]
    # plt.scatter(x, y)
    # plt.plot(n, m, 'r')
    # plt.show()


gradientDescent(x, y)
normalEquation(x, y)

# 感觉梯度下降法还是有点问题，theta_0 偏差有点大，可能是需要特征缩放
