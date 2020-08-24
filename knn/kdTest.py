import numpy as np

from knn.kd_node import Node

if __name__ == '__main__':
    T = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    sort = T.argsort(axis=0)
    head = Node(None,0,None)
    node = Node(T,1,head)
    node.init()

