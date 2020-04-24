import math
from matplotlib import pyplot as plt

# 阶乘
def func(n):
    if n == 0 or n == 1:
        return 1
    else:
        return (n * func(n - 1))


# 排列的可能性
def permutation(a, b):
    return func(a) / func(a - b)


num = [i for i in range(100)]
probablity = [1 - permutation(365, i) / math.pow(365, i) for i in range(100)]
# for i in range(100):
#     num.append(i)
#     a = 1 - permutation(365, i) / math.pow(365, i)
#     probablity.append(a)

# probablity = [1- (permutation(365, i) / math.pow(365, i)) for i in num]

plt.scatter(num, probablity)
plt.show()
