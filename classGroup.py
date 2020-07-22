# 组合：在一个类中以另外一个列的对象作为数据属性（一个类的属性是另一个类的对象）
class Equip:
    def fire(self):
        print("Fire")
class Riven:
    camp='Noxus'
    def __init__(self,name):
        self.name = name
        self.equip = Equip() # 用Equip类产生一个装备，赋值给equip属性

r1 = Riven('公孙离')
r1.equip.fire()

# 继承：建立派生类与基类之间的关系，是一种‘是’的关系
# 组合：建立类与组合之间的关系，是一种‘有’的关系

class People:
    def __init__(self,name,age,sex):
        self.name=name
        self.age=age
        self.sex=sex

class Course:
    def __init__(self,name,period,price):
        self.name=name
        self.period=period
        self.price=price
    def tell_info(self):
        print('<%s %s %s>' %(self.name,self.period,self.price))

class Teacher(People):
    def __init__(self,name,age,sex,job_title):
        People.__init__(self,name,age,sex)
        self.job_title=job_title
        self.course=[]
        self.students=[]


class Student(People):
    def __init__(self,name,age,sex):
        People.__init__(self,name,age,sex)
        self.course=[]

Ton = Teacher('Ton',18,'male','猎头老师')
s1 = Student('jery',18,'female')

python = Course('python','3min',300)
linux = Course('linus','8min','300.0')
# 为老师和学生添加课程
Ton.course.append(python)
Ton.course.append(linux)
s1.course.append(python)

# 为老师添加学生
Ton.students.append(s1)

for obj in Ton.course:
    obj.tell_info()
