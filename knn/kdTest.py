import numpy as np

from knn.kd_node import Node


def initKdTree(allPoints, level, parent):
    print("allPoints\n" + str(allPoints))
    curNode = Node(allPoints, level, parent)
    if (allPoints.shape[0] == 1):
        curNode.point = allPoints[0];
        return curNode

    var = np.var(allPoints, axis=0)
    dividDim = np.argsort(var)[var.shape[0] - 1]
    sort = np.argsort(allPoints, axis=0)[:, dividDim]
    l = int(len(sort) / 2)
    middleP = allPoints[sort[l], :]
    curNode.point = middleP

    if (l > 0):
        leftIndex = sort[0:l]
        leftData = allPoints[leftIndex, :]
        curNode.left = initKdTree(leftData, level + 1, curNode)

    if (len(sort) > l + 1):
        rightIndex = sort[l + 1:]
        rightData = allPoints[rightIndex]
        curNode.right = initKdTree(rightData, level + 1, curNode)
    return curNode


def printNode(node):
    parent = node.parent
    p = 'top' if parent == None else str(parent.point)
    print(str(node.point) + " level: " + str(node.level) + " parent: " + p)
    if (node.left != None):
        printNode(node.left)
    if (node.right != None):
        printNode(node.right)


if __name__ == '__main__':
    T = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])

    sort = T.argsort(axis=0)
    # head = Node(None, 0, None)
    # node = Node(T, 1, head)
    # node.init()
    head = initKdTree(T, 1, None)
    print(head)
    printNode(head)
