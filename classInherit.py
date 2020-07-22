# pyhton 中类可以继承一个或者多个父类
class ParentClass1:
    pass

class ParentClass2:
    pass

class SubClass1(ParentClass1): # 单继承
    pass

class SubClass2(ParentClass1,ParentClass2): # 多继承，逗号分割
    pass

# 查看继承
# _base_ 只查看从左到右继承的第一个子类
# _bases_ 查看所有继承的父类
print(ParentClass1.__base__,type(ParentClass1.__base__))
print(SubClass1.__base__,type(SubClass1.__base__))

print(SubClass2.__bases__) # 数据结构为元组

# python2中有经典类和新式类  python3中只有新式类
# python3中如果没有指定基类，都会默认继承object类，object类是所有python类中的基类


# 继承与抽象
# 抽象：通过抽象得到类
# 继承：基于抽象的结果，通过编程语言实现，先抽象，再经过继承表达
class Hero:
    def __init__(self,name,value,aggress):
        self.name = name
        self.value = value
        self.aggress = aggress
    def attack(self,enemy):
        self.value -= self.aggress
class Garen(Hero):
    pass
class Riven(Hero):
    pass
g1 = Garen('绿巨人',80,20)
# __dict__:类的静态函数、类函数、普通函数、全局变量以及一些内置的属性都是放在类__dict__里
print(g1.__dict__)

r1 = Riven('钢铁侠',200,34)
print(g1.value)
r1.attack(g1) # 疑惑区
print(g1.value)


# 属性查找
class Foo:
    def f1(self):
        print("Foo.f1")
    def f2(self):
        print("Foo.f2")
class Bar(Foo):
    def f2(self):
        print("Bar.f2")
b = Bar()
print("0",b.__dict__)
print("1",b.f1()) # 调用父类中的
print("2",b.f2())  # 调用自己的

# 派生
# 父类和子类都有的方法叫做方法的重写
# 父类里没有子类有的方法叫派生方法
class Riven(Hero):
    camp='Noxus'
    def __init__(self,nickname,aggressivity,life_value,skin):
        Hero.__init__(self,nickname,aggressivity,life_value) #调用父类功能
        self.skin=skin #新属性
    def attack(self,enemy): #在自己这里定义新的attack,不再使用父类的attack,且不会影响父类
        Hero.attack(self,enemy) #调用功能
        print('from riven')
    def fly(self): #在自己这里定义新的
        print('%s is flying' %self.nickname)

r1=Riven('锐雯雯',57,200,'皮皮')
r1.fly()
print(r1.skin)

