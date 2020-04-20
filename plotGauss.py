import math

import matplotlib.pyplot as plt
import numpy as np

# 画高斯分布的概率函数图
t = np.arange(-10, 10, 0.01)

u = 0
c2 = 5
y = [1 / ((math.sqrt(2 * math.pi * c2))) * math.exp(-1 * math.pow(i - u, 2) / (2 * c2)) for i in t]

plt.plot(t, y, '-')
plt.show()
