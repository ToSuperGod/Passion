# 隐藏属性
class A:
    __x = 1  # 双下划线开头隐藏属性
    def __init__(self,name):
        self.name = name
    def __foo(self):
        # print(type(self.name))
        print("%s foo run" % self.name)
    def bar(self):
        self.__foo()
        print("from bar")

print(A.__dict__)

a = A('egon')
a._A__foo() # 可以访问隐藏属性
a.bar()

# 在继承中，父类如果不想让子类覆盖自己的方法，可以将方法定义为私有的


# 要实现隐藏属性赋值，需要运用@name.setter装饰器(name被property装饰后才可用)
class People:
    def __init__(self,name):
        self.__name = name

    @property
    def name(self):
        return self.__name

    @name.setter  # 必须这么修改
    def name(self,val):
        if not isinstance(val,str):
            print("名字必须是字符串类型")
            return
        self.__name = val

    @name.deleter  # 删除
    def name(self):
        print("不许删除")

p = People('Tom')
print(p.name)
p.name = 'Jeri'
print(p.name)
p.name = 123
del p.name
# 实现了name属性的修改和删除，@name.deleter和@name.setter都是基于name被@property装饰才可用的。




class Foo:
    def __init__(self, name):
        self.name = name

    def tell(self):  # 绑定了对象的函数
        print('名字是%s' % self.name)

    @classmethod
    def func(cls):  # 绑定了类的方法，默认传入参数是类本身
        print(cls)

    @staticmethod
    def func1(x, y):   # 在类内部的普通函数
        print(x+y)

f = Foo('egon')
print(Foo.tell)
print(f.tell)