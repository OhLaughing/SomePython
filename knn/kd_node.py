import numpy as np
class Node:
    def __init__(self, allPoints, level):
        self.allPoints = allPoints
        self.level = level
        self.point = 0
        self.left = 0
        self.right = 0
        self.parent
        self.pointIndex = []

    def constructTree(self):
        sort = np.argsort(self.allPoints)
        var = np.var(self.allPoints, axis=0)
        dividDim = np.argsort(var)[var.shape[0]-1]
        sort = np.argsort(self.allPoints, axis=0)[:,dividDim]
        middleP = self.allPoints
        # 在一个维度上分割数据之后，下一次分割的维度怎么算（要不要还算父分割的维度）
        print(dividDim)
