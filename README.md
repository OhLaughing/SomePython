# 所有代码都是python3
## python
  机器学习连接：https://www.cnblogs.com/aabbcc/p/8683042.html
- 【数学】拉格朗日对偶，从0到完全理解：https://blog.csdn.net/frostime/article/details/90291392
- global set value
···python
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
···

## paper
- A practical guide to SVM classification 


