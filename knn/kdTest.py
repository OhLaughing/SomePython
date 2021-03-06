import operator

import numpy as np

import knn.utils as utils
from knn.kd_node import Node


# allPoints: 所有的训练数据
# level: 根节点的level为1，越往叶子节点level越大
# parent： 父节点
# dividDim: 切分的轴
def initKdTree(allPoints, level, parent, dividDim):
    m = allPoints.shape[1] - 1
    curNode = Node(level, parent)
    if (allPoints.shape[0] == 1):
        data = allPoints[0]
        curNode.point = data[:-1]
        curNode.index = int(data[-1])
        return curNode
    #
    # var = np.var(allPoints[:, :m - 1], axis=0)
    # dividDim = np.argsort(var)[var.shape[0] - 1]
    curNode.dividDim = dividDim
    sort = np.argsort(allPoints, axis=0)[:, dividDim]
    l = int(len(sort) / 2)
    middleP = allPoints[sort[l], :]
    curNode.point = middleP[:-1]
    curNode.index = int(middleP[-1])
    nextDividDim = dividDim + 1
    if (nextDividDim == m):
        nextDividDim = 0

    if (l > 0):
        leftIndex = sort[0:l]
        leftData = allPoints[leftIndex, :]
        curNode.left = initKdTree(leftData, level + 1, curNode, nextDividDim)

    if (len(sort) > l + 1):
        rightIndex = sort[l + 1:]
        rightData = allPoints[rightIndex]
        curNode.right = initKdTree(rightData, level + 1, curNode, nextDividDim)
    return curNode


def printNode(node):
    parent = node.parent
    p = 'top' if parent == None else str(parent.point)
    print(str(node.point) + " level: " + str(node.level) + " parent: " + p + " index: " + str(node.index) +
          " dividDim: " + str(node.dividDim))
    if (node.left != None):
        printNode(node.left)
    if (node.right != None):
        printNode(node.right)


def getLeafNode(node, x):
    # node.dividDim==None 就是叶子节点
    currNode = node
    while (currNode != None and currNode.dividDim != None):
        point = currNode.point

        if (x[currNode.dividDim] > point[currNode.dividDim]):
            currNode = currNode.right if currNode.right != None else currNode.left
        else:
            currNode = currNode.left if currNode.left != None else currNode.right
    return currNode


# 该方法用在从兄弟节点到叶子节点，找到是否有离目标点更近的
def findToLeaf(node, x, nearestNode, nearestDistant):
    while (node != None):
        distant = utils.getDistant(node.point, x)
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


# 把NodeDistant插入到list中
def insertToList(nearestkNodeList, nodeDistant):
    index = 0
    while (index < len(nearestkNodeList) and nearestkNodeList[index].distant > nodeDistant.distant):
        index += 1
    nearestkNodeList.insert(index, nodeDistant)


def findToLeaf_k(node, x, nearestkNodeList, k):
    head = node
    current = getLeafNode(head, x)
    while (current != head):
        distant = utils.getDistant(node.point, x)
        if (len(nearestkNodeList) < k):
            insertToList(nearestkNodeList, NodeDistant(node, distant))
        elif distant < nearestkNodeList[0].distant:
            assert len(nearestkNodeList) == k
            nearestkNodeList.remove(nearestkNodeList[0])
            insertToList(nearestkNodeList, NodeDistant(node, distant))
        dim = node.dividDim

        if (dim == None):
            node = None
        elif (x[dim] > node.point[dim]):
            node = node.right if node.right != None else node.left
        else:
            node = node.left if node.left != None else node.right


def findNearestNode(node, x):
    # 先找到叶子节点
    leafNode = getLeafNode(node, x)
    nearestNode = leafNode
    currentNode = leafNode
    nearestDistant = utils.getDistant(leafNode.point, x)
    print(nearestDistant)
    parentNode = leafNode.parent
    while parentNode != None:
        parentDividDim = parentNode.dividDim
        # 以x为中心，以当时找到的最近距离为半径做圆，查看该圆是否与父节点的分隔线有相交
        if (nearestDistant > np.abs(parentNode.point[parentDividDim] - x[parentDividDim])):
            # 从兄弟节点开始
            siblingNode = parentNode.left if currentNode == parentNode.right else parentNode.right

            nearestNode, nearestDistant = findToLeaf(siblingNode, x, nearestNode, nearestDistant)

        disTant2 = utils.getDistant(parentNode.point, x)
        if (disTant2 < nearestDistant):
            nearestDistant = disTant2
            nearestNode = parentNode
        tmp = parentNode
        parentNode = parentNode.parent
        currentNode = tmp

    return nearestNode


class NodeDistant:
    def __init__(self, node, distant):
        self.node = node
        self.distant = distant

    def __str__(self):
        return '{}-distant: {}'.format(self.node, self.distant)


def findkNearestNode(nearestkNodeList, node, x, k):
    # 先找到叶子节点
    leafNode = getLeafNode(node, x)
    # 此时，最近的点，及最近距离指的是，k个中距离最大的那个
    currentNode = leafNode
    distant = utils.getDistant(leafNode.point, x)

    if len(nearestkNodeList) < k:
        insertToList(nearestkNodeList, NodeDistant(currentNode, distant))
    elif distant < nearestkNodeList[0].distant:
        assert len(nearestkNodeList) == k
        nearestkNodeList.remove(nearestkNodeList[0])
        insertToList(nearestkNodeList, NodeDistant(currentNode, distant))

    parentNode = leafNode.parent
    while currentNode != node:
        parentDividDim = parentNode.dividDim
        # 以x为中心，以当时找到的最近距离为半径做圆，查看该圆是否与父节点的分隔线有相交
        if nearestkNodeList[0].distant > np.abs(parentNode.point[parentDividDim] - x[parentDividDim]) or len(
                nearestkNodeList) < k:
            # 从兄弟节点开始
            siblingNode = parentNode.left if currentNode == parentNode.right else parentNode.right
            if (siblingNode != None):
                findkNearestNode(nearestkNodeList, siblingNode, x, k)
                # findToLeaf_k(siblingNode, x, nearestkNodeList, k)

        disTant2 = utils.getDistant(parentNode.point, x)

        if (len(nearestkNodeList) < k):
            insertToList(nearestkNodeList, NodeDistant(parentNode, disTant2))

        elif (disTant2 < nearestkNodeList[0].distant):
            assert len(nearestkNodeList) == k
            nearestkNodeList.remove(nearestkNodeList[0])
            insertToList(nearestkNodeList, NodeDistant(parentNode, disTant2))

        tmp = parentNode
        parentNode = parentNode.parent
        currentNode = tmp

    return nearestkNodeList


def getNearestLabel(nodeList, labels):
    labesDict = {}
    for i in range(len(nodeList)):
        label = labels[int(nodeList[i].node.index)]
        labesDict[label] = labesDict.get(label, 0) + 1
    sortedClassCount = sorted(labesDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


if __name__ == '__main__':
    T = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])

    sort = T.argsort(axis=0)
    # head = Node(None, 0, None)
    # node = Node(T, 1, head)
    # node.init()
    T = np.c_[T, np.arange(T.shape[0])]
    head = initKdTree(T, 1, None, 0)
    print(head)
    printNode(head)
    x = np.array([6.5, 1])
    nearestNode = findNearestNode(head, x)
    print(nearestNode.point)

    # n = findkNearestNode(head, x, 1)
    # for i in range(len(n)):
    #     print(str(n[i]))
