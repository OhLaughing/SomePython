import matplotlib.pyplot as plt
import numpy as np


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
    iter = 100
    rate = 0.0001
    X = np.c_[np.ones(len), x]
    w = np.random.random(2)
    for i in range(iter):
        C = (X.dot(w) - y).T.dot(X.dot(w) - y)
        print("cost function: " + str(C))

        dw = X.T.dot(X.dot(w) - y)
        w = w - (dw * rate)
    return w

def plotLine(w):
    x = np.linspace(0,20,10)
    y = x*w[1]+w[0]
    plt.plot(x,y,'-r')

if __name__ == '__main__':
    x, y = loadData('ex1data1.txt')
    len = len(x)
    x = np.array(x)
    y = np.array(y)
    w = gradientDescent(x, y)
    plotData(x,y)
    plotLine(w)
    plt.show()