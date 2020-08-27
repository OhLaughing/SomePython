import numpy as np


class Node:
    def __init__(self, level, parent):
        self.level = level
        self.parent = parent
        self.index = -1
        self.point = None
        self.left = None
        self.right = None
        self.dividDim = None  # 切分维度

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

    def __str__(self):
        return 'point:{}-level:{}-divide dimension:{}-index:{}'.format(self.point, self.level, self.dividDim,
                                                                       self.index)
