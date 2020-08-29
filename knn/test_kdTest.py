from unittest import TestCase

from knn.kdTest import *
import knn.utils as utils

class Test(TestCase):
    def test_init_kd_tree(self):
        T = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
        T = np.c_[T,np.arange(T.shape[0])]
        head = initKdTree(T, 1, None, 0)
        print(head)
        printNode(head)
        x = np.array([6.5, 1])
        nearestNode = findNearestNode(head, x)
        print(nearestNode.point)

        assert (np.array([7, 2]) == nearestNode.point).all()

        x = np.array([6.5, 0])
        nearestNode = findNearestNode(head, x)
        assert (np.array([8, 1]) == nearestNode.point).all()
        x = np.array([4, 3])
        nearestNode = findNearestNode(head, x)
        assert (np.array([5, 4]) == nearestNode.point).all()

        x = np.array([3, 5])
        nearestNode = findNearestNode(head, x)
        # [3, 5] 与[2, 3]5、[5, 4]、[4, 7]距离相同
        assert (np.array([4, 7]) == nearestNode.point).all()

        x = np.array([5 ,6])
        nearestNode = findNearestNode(head, x)
        assert (np.array([4, 7]) == nearestNode.point).all()

    def test_init_kd_tree1(self):
        a = np.array([2, 3])
        b = np.array([5, 4])
        c = np.array([3, 5])
        print(utils.getDistant(a, c))
        print(utils.getDistant(b, c))
