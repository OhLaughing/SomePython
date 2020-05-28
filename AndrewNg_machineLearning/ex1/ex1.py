import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def loadData(file):
    f = open(file, 'r')
    lines = f.readlines()
    x = []
    y = []
    for line in lines:
        line = line.strip('\n')
        v = line.split(',')
        x.append(float(v[0]))
        y.append(float(v[1]))
    return x, y


def plotData(x, y):
    plt.scatter(x, y, 15, 'r', 'x')
    plt.xlabel('Population of City in 10000s')
    plt.ylabel('Profit in $10000s')


def gradientDescent(x, y):
    iter = 2000
    rate = 0.0003
    X = np.c_[np.ones(len), x]
    w = np.array([0,0])
    for i in range(iter):
        C = (X.dot(w) - y).T.dot(X.dot(w) - y)/(2*97)
        print("cost function: " + str(C))

        dw = X.T.dot(X.dot(w) - y)/97
        w = w - (dw * rate)
    return w

def plotLine(w):
    x = np.linspace(0,20,10)
    y = x*w[1]+w[0]
    plt.plot(x,y,'-r')

def plotcostFunction(x,y):
    fig1 = plt.figure()
    ax = Axes3D(fig1)
    X = np.c_[np.ones(len), x]
    w0, w1 = np.mgrid[-2:2:40j, -2:2:40j]
    w = np.array([w0,w1])
    J=0
    for i in range(97):
        J+=(w0+w1*x[i]-y[i])**2

    plt.title("This is main title")  # 总标题
    ax.plot_surface(w0, w1, J, rstride=1, cstride=1, cmap=plt.cm.coolwarm, alpha=0.5)  # 用取样点(x,y,z)去构建曲面
    ax.set_xlabel('x label', color='r')
    ax.set_ylabel('y label', color='g')
    ax.set_zlabel('z label', color='b')  # 给三个坐标轴注明
    plt.show()  # 显示模块中的所有绘图对象

if __name__ == '__main__':
    x, y = loadData('ex1data1.txt')
    len = len(x)
    x = np.array(x)
    y = np.array(y)
    plotcostFunction(x,y)
    # w = gradientDescent(x, y)
    # X = np.c_[np.ones(len), x]
    # C = (X.dot(w) - y).T.dot(X.dot(w) - y)/(2*97)
    # for i in range(100):
    #     C1 = (X.dot(w) - y).T.dot(X.dot(w) - y)/(2*97)
    #     print('C: ' + str(C1))
    #     if(C1 > C):
    #         break
    #     C = C1
    #     w[0]+=0.01
    # plotData(x,y)
    # print(w)
    # plotLine(w)
    # plt.show()