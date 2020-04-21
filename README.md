# 所有代码都是python3
## python
  机器学习连接：https://www.cnblogs.com/aabbcc/p/8683042.html
- 【数学】拉格朗日对偶，从0到完全理解：https://blog.csdn.net/frostime/article/details/90291392
- global set value
```python
x = 1
y = 2
x, y = swap(x, y)     # => x = 2, y = 1
# (x, y) = swap(x,y)  # Again parenthesis have been excluded but can be included.

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
```python
L = list(range(10))
L2 = [str(c) for c in L]
```


## paper
- A practical guide to SVM classification 

## numpy
- np.ones((3,5))、np.full((3,5), 3.14)、
- np.linspace(0,1,10) 生成等差数列，start：0,stop:1,共10个数（包括start和stop）
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
