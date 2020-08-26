from knn.kd_node import Node
from knn.kdTest import *
import numpy as np
def test_init_kd_tree():
    T = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])

    sort = T.argsort(axis=0)
    # head = Node(None, 0, None)
    # node = Node(T, 1, head)
    # node.init()
    head = initKdTree(T, 1, None)
    print(head)
    printNode(head)
    x = np.array([6.5, 1])
    nearestNode = findNearestNode(head, x)
    print(nearestNode.point)

    assert (np.array([7,2]) == nearestNode.point).all()
