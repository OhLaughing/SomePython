# 所有代码都是python3
## python
  机器学习连接：https://www.cnblogs.com/aabbcc/p/8683042.html
- 【数学】拉格朗日对偶，从0到完全理解：https://blog.csdn.net/frostime/article/details/90291392
- global set value
- 机器学习撸一遍：https://www.zhihu.com/question/326770694/answer/715217571

## 链接
- [the matrix cookbook](http://www2.imm.dtu.dk/pubdb/edoc/imm3274.pdf)
- 深度学习书籍推荐：https://zhuanlan.zhihu.com/p/84920970
-《神经网络与深度学习》：https://tigerneil.gitbooks.io/neural-networks-and-deep-learning-zh/content/
- scikit learn 初探35题，matplotlib50题，pandas 50题：https://zhuanlan.zhihu.com/p/97977032
- 深度学习四大名著之《Scikit-Learn、Keras与TensorFlow机器学习实用指南（第二版）》 https://www.jianshu.com/p/4a94798f7dcc
- [keras官方example](https://keras.io/examples/)
- [keras中文文档](https://keras-cn.readthedocs.io/en/latest/)
- <https://tensorflow.google.cn/tutorials/keras/classification>
```python
x = 1
y = 2
x, y = swap(x, y)     # => x = 2, y = 1
# (x, y) = swap(x,y)  # Again parenthesis have been excluded but can be included.
# python中with的作用 : https://zhuanlan.zhihu.com/p/158100802

# Function Scope
x = 5

def set_x(num):
    # Local var x not the same as global variable x
    x = num    # => 43
    print(x)   # => 43

def set_global_x(num):
    global x
    print(x)   # => 5
    x = num    # global var x is now set to 6
    print(x)   # => 6

set_x(43)
set_global_x(6)
```
- python命令，who和whos可以查看所有变量，以及变量的信息，利用del命令可以删除变量的定义
- 会利用 for a in b
- python3中/表示浮点数除法，//表示整数除法
```python
L = list(range(10))
L2 = [str(c) for c in L]
```
sorted和operator.itemgetter实现排序，也可以通过加reverse=True来实现逆序排序
```python
import operator
students = [('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10)]
sorted(students, key=operator.itemgetter(2))  # 根据第3个值进行排序
# [('dave', 'B', 10), ('jane', 'B', 12), ('john', 'A', 15)]
sorted(students, key=operator.itemgetter(1,2)) # 根据第二个和第三个值进行排序
# [('john', 'A', 15), ('dave', 'B', 10), ('jane', 'B', 12)]
```
- Stochastic Gradient Descent (SGD)
- CS229的课程大纲（包含习题和课题讨论资料）：http://cs229.stanford.edu/syllabus-autumn2018.html
- 在D:\workspace\learnprojects\SomePython-master\目录进入python解释器，import AndrewNg_machineLearning.ex2.ex2 as ex2
然后 执行ex2.test1() 即可调用test1()方法
- 神经网络基础[https://www.cnblogs.com/maybe2030/p/5597716.html]
- 查看已定义所有变量命令：dir()， 删除变量命令del

## paper书籍
- A practical guide to SVM classification 
《DEEP LEARNING》《机器学习-周志华》《统计学习方法-李航》《机器学习实战》《利用Python进行数据分析》

## 学习记录
- 20200817： 前段时间一直在看吴恩达的机器学习课程，共8课，看到第4课神经网络，还没能自己通过反向传播来实现把例子的，手写数字识别，代码中不知道哪里还有些问题，昨天看了《机器学习实战》的knn算法，还没看完
- 20200822： 看完《机器学习实战》的knn算法，在看《统计学习方法》的KNN，这里用KD树算法，用KD的主要目的是，找到最近的距离时，用二分查找方法，否则要遍历所有的样本才能知道最近的样本距离，效率很低，总结就是KD树查找效率为O(lg(n)),不用KD树（机器学习实战里的方法）效率为O(n)

## ipython
- ?\??都是查询方法或模块的信息，例如：np.arange?\np.arange??, ??比？更详细，也可以help(np.arange)查看信息
- tab键，可以自动补全，例如，输入np.ar+tab，就会出现np.ar开头的所有的方法
- str.*find*? 可以查找方法
- 查看历史命令，可以用上下键，也可以用ctrl+p/ctrl+n, 但是用ctrl的好处是，可以指定开头，比如我之前输入了很多命令，但是我只想查看np.开头的命，就可以输入np.之后，通过ctrl+p/ctrl+n来查找
- 可以在ipython中，使用ls、pwd、cd等命令
- 在python3中，reload，可以先import importlib 然后 importlib.reload(kNN1) 

## numpy
- np.ones((3,5))、np.full((3,5), 3.14)、
- np.linspace(0,1,10) 生成等差数列，start：0,stop:1,共10个数（包括start和stop），num默认50  endpoint 设为 true或False来控制是否包含stop
- np.array([1,2,3])
- n.arange(0,10，2) 生成等差数列，包括start，不包括stop
- 数组的属性：ndim、shape、size、dtype
- x[-1]取数组末尾数据
- 多维数组，x[1,2]取1行2列元素
- 与python不同，numpy 是固定类型的
- x[start:stop:step]取元素，默认start=0,stop为数组维度，step=1
- x[::2] 从0每隔一个取一个元素，x[1::2] 从1每隔一个取一个元素,x[::-1]逆序取元素
- x2是二维的，x2[:3,::2]取前3行，列从0开始每隔一个取一个元素
- x2[0]等同于x2[0,:]
- grid = np.arange(1,10).reshape(3,3)
- np.flatten() 把多维数组变成一维
- np.dot(a,b) 如果a和b是向量，结果是内积，如果a和b是矩阵，结果是矩阵积，也可以写成a.dot(b)
- np.tile() 函数，扩展数组
- numpy的广播功能
```python
a = np.arange(3)
b = np.arange(3).reshape(3,1)
In [12]: a*b
Out[12]:
array([[0, 0, 0],
       [0, 1, 2],
       [0, 2, 4]])
```
- a = (x for x in range(5))得到generator对象，然后通过a.next()获取元素，会提示： 'generator' object has no attribute 'next'，应该用a.__next__()
- 笛卡尔坐标、极坐标
- np.c_ 是将两个矩阵左右组合到一个矩阵（c为column）,用法np.c_[a,b],不是括号
- np.r_ 是将两个矩阵上下组合到一个矩阵（r为row）
- ravel 将多维数组转成一维数组
- numpy.ndarray可以用.transpose()方法来转置
- 在ipython中如果想知道某个方法的详细信息说明，可以用help命令，例如：help(np.ravel)即可
- 二维numpy矩阵，a[0]和a[0,:]都是第一行
- np.info（）方法是获取其他方法的详细信息
