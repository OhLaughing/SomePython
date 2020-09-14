# 通过 方法名.全局变量 的方式来定义方法内的全局变量，该变量都可以访问
def a():
    a.tt = 'world'
def b():
    a.tt = 'hello'
def c():
    print(a.tt)

b()
a()
c()
