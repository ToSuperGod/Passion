d = dict({'name':'egon'})
print(isinstance(d,dict))
print(d)
中国 = "China"
print(中国)

class Foo:
    def __init__(self,name):
        self.name = name
    def __getitem__(self, item):
        return self.__dict__.get(item)
    def __setitem__(self, key, value):
        self.__dict__[key] = value
    def __delitem__(self, key):
        del self.__dict__[key]

# 查看属性
obj = Foo('egon')
print(obj['name'])

# 设置属性
obj['sex'] = 'male'
print(obj.sex)


# 删除属性
del obj['name']
print(obj.__dict__)


# 改变字符串的显示  __str__,__repr__
# 自定制格式化字符串__format__
d = dict({'name':'egon'})
print(isinstance(d,dict)) # d 是 dict类的实例
print(d)

class Foo:
    pass
obj = Foo()
print(obj)

# __str__方法定义后，会在打印对象的收，把字符串的的结果作为打印的结果
class People:
    def __init__(self,name,age):
        self.name = name
        self.age = age
    def __str__(self): # 必须返回一个字符串
        return '<name:%s,age:%s>' % (self.name,self.age)

obj = People('egon',18)
print(obj)

format_dict={
    'nat':'{obj.name}-{obj.addr}-{obj.type}',# 学校名-地址-类型
    'tna':'{obj.type}:{obj.name}:{obj.addr}',# 类型：名：地址
    'tan':'{obj.type}/{obj.addr}/{obj.name}',# 类型/地址/名
}
class School:
    def __init__(self,name,addr,type):
        self.name = name
        self.addr = addr
        self.type = type
    def __repr__(self):
        return 'School(%s,%s)' %(self.name,self.addr)
    def __str__(self):
        return  '(%s,%s)' %(self.name,self.addr)
    def __format__(self, format_spec):
        if not format_spec or format_spec not in format_dict:
            format_spec='nat'
        fmt = format_dict[format_spec]
        return fmt.format(obj=self)
s1 = School('oldboy1','北京','私立')
print('from repr:',repr(s1))
print('from str:',str(s1))
print(s1)

print(format(s1,'nat'))
print(format(s1,'tna'))
print(format(s1,'asfdasdffd'))



# issubclass和isinstance
class A:
    pass

class B(A):
    pass

print(issubclass(B,A)) #B是A的子类,返回True

a1=A()
print(isinstance(a1,A)) #a1是A的实例



