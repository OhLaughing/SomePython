import numpy as np

from knn.kd_node import Node


def initKdTree(allPoints, level, parent):
    print("allPoints\n" + str(allPoints))
    curNode = Node(allPoints, level, parent)
    if (allPoints.shape[0] == 1):
        curNode.point = allPoints[0]
        return curNode

    var = np.var(allPoints, axis=0)
    dividDim = np.argsort(var)[var.shape[0] - 1]
    curNode.dividDim = dividDim
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
    print(str(node.point) + " level: " + str(node.level) + " parent: " + p + " dividDim: " + str(node.dividDim))
    if (node.left != None):
        printNode(node.left)
    if (node.right != None):
        printNode(node.right)


def getLeafNode(node, x):
    # node.dividDim==None 就是叶子节点
    if (node.dividDim == None):
        return node
    point = node.point
    if (x[node.dividDim] > point[node.dividDim]):
        return getLeafNode(node.right, x)
    else:
        return getLeafNode(node.left, x)


def getDistant(x, y):
    dif = x - y
    square = dif ** 2
    sum = np.sum(square)
    return sum ** 0.5


# 该方法用在从兄弟节点到叶子节点，找到是否有离目标点更近的
def findToLeaf(node, x, nearestNode, nearestDistant):
    while (node != None):
        distant = getDistant(node.point, x)
        if (distant < nearestDistant):
            nearestDistant = distant
            nearestNode = node
        dim = node.dividDim
        if (dim == None):
            node = None
        elif (x[dim] > node.point[dim]):
            node = node.right
        else:
            node = node.left
    return nearestNode, nearestDistant


def findNearestNode(node, x):
    # 先找到叶子节点
    leafNode = getLeafNode(node, x)
    nearestNode = leafNode
    currentNode = leafNode
    nearestDistant = getDistant(leafNode.point, x)
    print(nearestDistant)
    parentNode = leafNode.parent
    while parentNode != None:
        parentDividDim = parentNode.dividDim
        # 以x为中心，以当时找到的最近距离为半径做圆，查看该圆是否与父节点的分隔线有相交
        if (nearestDistant > np.abs(parentNode.point[parentDividDim] - x[parentDividDim])):
            # 从兄弟节点开始
            siblingNode = parentNode.left if currentNode == parentNode.right else parentNode.right

            nearestNode, nearestDistant = findToLeaf(siblingNode, x, nearestNode, nearestDistant)

        disTant2 = getDistant(parentNode.point, x)
        if (disTant2 < nearestDistant):
            nearestDistant = disTant2
            nearestNode = parentNode
        tmp = parentNode
        parentNode = parentNode.parent
        currentNode = tmp

    return nearestNode


def findkNearestNode(node, x, k):
    # 先找到叶子节点
    nearestkNode = {}
    leafNode = getLeafNode(node, x)
    # 此时，最近的点，及最近距离指的是，k个中距离最大的那个
    nearestNode = leafNode
    currentNode = leafNode
    nearestDistant = getDistant(leafNode.point, x)
    nearestkNode[currentNode.point] = nearestDistant

    print(nearestDistant)
    parentNode = leafNode.parent
    while parentNode != None:
        parentDividDim = parentNode.dividDim
        # 以x为中心，以当时找到的最近距离为半径做圆，查看该圆是否与父节点的分隔线有相交
        if (nearestDistant > np.abs(parentNode.point[parentDividDim] - x[parentDividDim])):
            # 从兄弟节点开始
            siblingNode = parentNode.left if currentNode == parentNode.right else parentNode.right

            nearestNode, nearestDistant = findToLeaf(siblingNode, x, nearestNode, nearestDistant)

        disTant2 = getDistant(parentNode.point, x)
        if (disTant2 < nearestDistant):
            nearestDistant = disTant2
            nearestNode = parentNode
        tmp = parentNode
        parentNode = parentNode.parent
        currentNode = tmp

    return nearestNode


if __name__ == '__main__':
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
