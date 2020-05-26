import numpy as np
import matplotlib.pyplot as plt

def loadData(file):
    f = open(file, 'r')
    lines= f.readlines()
    x=[]
    y=[]
    for line in lines:
        line = line.strip('\n')
        v = line.split(',')
        x.append(float(v[0]))
        y.append(float(v[1]))
    return x,y



if __name__ == '__main__':
    x,y =loadData('ex1data1.txt')
    print(len(x))
    print(x)
    print(type(x[0]))
    print(len(y))
    plt.scatter(x,y, 15,'r', 'x')
    plt.show()