import numpy as np


class Node:
    def __init__(self, allPoints, level, parent):
        self.allPoints = allPoints
        self.level = level
        self.parent = parent
        self.point = 0
        self.left = 0
        self.right = 0

    def init(self):
        sort = np.argsort(self.allPoints)
        var = np.var(self.allPoints, axis=0)
        dividDim = np.argsort(var)[var.shape[0] - 1]
        sort = np.argsort(self.allPoints, axis=0)[:, dividDim]
        l = int(len(sort) / 2)
        middleP = self.allPoints[sort[l]]
        self.point = middleP
        self.left = Node(self.allPoints, self.level + 1, self)
        self.right = Node(self.allPoints, self.level + 1, self)
        self.left.init()
        self.right.init()
        # 在一个维度上分割数据之后，下一次分割的维度怎么算（要不要还算父分割的维度）
        print(dividDim)
