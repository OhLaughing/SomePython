from sklearn.datasets import make_blobs
from matplotlib import pyplot

data, label = make_blobs(n_samples=100, n_features=2, centers=5)

# 绘制样本显示
pyplot.scatter(data[:, 0], data[:, 1], c=label)
pyplot.show()

from sklearn import linear_model

reg = linear_model.LinearRegression()
reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])