#  抽象类只能被继承，不能被实例化
# 抽象类主要用来进行类型隐藏和充当全局变量的角色
# python 借用abc模块实现抽象类和接口
import abc
class Animal(metaclass=abc.ABCMeta):
    @abc.abstractclassmethod
    def run(self):
        pass
    @abc.abstractclassmethod
    def eat(self):
        pass
class prople(Animal):
    def run(self):
        print("people is running")
    def eat(self):
        print("prople is eatting")

p = prople()
p.run()