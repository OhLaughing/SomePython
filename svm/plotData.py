from svm.SVM import *
import numpy as np
from matplotlib import pyplot as plt
data,label = loadDataSet('testSet.txt')

narray = np.array(data)
good=[]
bad=[]
x=narray[:,0]
y=narray[:,1]
for i in range(0,len(label)):
    if(label[i]==1):
        good.append([x[i], y[i]])
    elif(label[i]==-1):
        bad.append([x[i], y[i]])
good=np.array(good)
bad=np.array(bad)

plt.title('SVM example show')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(good[:,0], good[:,1], 'ro', bad[:,0], bad[:,1], 'b+')
plt.show()
