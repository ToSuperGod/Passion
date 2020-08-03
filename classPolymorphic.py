# 多太目的是实现接口重用
import abc
class Animal(metaclass=abc.ABCMeta):
    @abc.abstractclassmethod
    def talk(self):
        pass
class People(Animal):
    def talk(self):
        print("People is taking")
class Pig(Animal):
    def talk(self):
        print("Pig is taking")

# 动态多态，调用接口调用
def func(animal):
    animal.talk()
po = People()
pig = Pig()

func(po)
func(pig)

