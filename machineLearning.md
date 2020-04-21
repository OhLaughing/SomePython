# notes
- torch.rand从[0, 1)随机抽取均匀分布， torch.randn 标准正态分布
- torch和numpy转换， b=a.numpy(), b = torch.from_numpy(a)
- torch.mm(a,b) 是矩阵相乘，a的列必须和b的行数相等

- sklearn的train_test_split是能够根据分类进行均分，使train和test的数据中的比例相同，测试如下：
```python
x=np.arange(900)
y=np.zeros(900)
for i in range(300,600):
    y[i]=1
for i in range(600,900):
    y[i]=2

```
- 可以通过sum(y==0),sum(y==1)命令查看y中各个数据的比例，现通过train_test_split对样本数据进行分割
```python
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)
```
- 由于每类数据有300个，test_size=0.33,因此，分配到train和test的数据比例为2:1，即train为200左右，test为100左右，通过sum(y_train==0)\
sum(y_train==1) sum(y_test==0) 等命令查看可知，分配的策略是安装比例分割的
- sklearn自带数据介绍：https://blog.csdn.net/weixin_42039090/article/details/80614918
- sklearn 中文教程：https://www.kesci.com/home/project/5df87b152823a10036aca1a9
- scikit-learn (sklearn) 官方文档中文版：https://sklearn.apachecn.org/#/
- 如果a是numpy.ndarray，如果是一维的，a[:2]表示前两个元素，a[2:]表示从第二个到最后， 如果a是二维的，a[:2]表示前两行，a[2:]表示从第二行到最后，a[:, :2]表示取所有行，前两列

- 机器学习实战(用Scikit-learn和TensorFlow进行机器学习: https://me.csdn.net/fjl_CSDN
- 引入核函数本质：存在一些核函数，在不知道映射函数的情况下，能得到和使用映射函数一样的结果。而且使用核函数，能使计算更加简单，有效。
- SVM推导：https://blog.csdn.net/u011529752/article/details/64443133

```
线性代数概念Top 3：

1. 矩阵运算

2. 特征值/特征向量

3. 向量空间和范数

微积分概念Top 3：

1. 偏导数

2. 向量值函数

3. 方向梯度

统计概念Top 3：

1. 贝叶斯定理

2. 组合学

3. 抽样方法
```
- 提升机器学习数学基础，这7本书一定要读-附pdf资源：https://blog.csdn.net/xinshucredit/article/details/89552600

## 学习知识点记录
-   normalization\make_classification
-   RBF Kernel
-   cross-validation
-   svm 正则化参数
-   核技巧
-   凸优化，矩阵求导
-   极大似然估计
-   多维高斯分布
-   随机向量
-   线性代数：二次型
