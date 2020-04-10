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
可以通过sum(y==0),sum(y==1)命令查看y中各个数据的比例，现通过train_test_split对样本数据进行分割
```python
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)
```
由于每类数据有300个，test_size=0.33,因此，分配到train和test的数据比例为2:1，即train为200左右，test为100左右，通过sum(y_train==0)\
sum(y_train==1) sum(y_test==0) 等命令查看可知，分配的策略是安装比例分割的
- 
