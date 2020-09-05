# os Process build
import os
import time
import random


class BTree(object):
    def __init__(self, data):
        self.data = data
        self.son = None
        self.brother = None


class PCB():
    def __init__(self, pid=None, ppid=None, uid=None, general=[],
                 orderCount = None, PSW = None, stackPoint = None, stats=None,
                 priority=None,other=None,incident=[],location=None,
                 synCommunication=[],resource=[],listPoint=None):

        self.pid=pid
        self.ppid=ppid
        self.uid=uid

        self.general=general # 通用寄存器
        self.orderCount=orderCount
        self.PSW=PSW
        self.stackPoint=stackPoint

        self.stats=stats
        self.priority=priority
        self.other=other
        self.incident=incident

        self.location=location
        self.synCommunication=synCommunication
        self.resource=resource
        self.listPoint=listPoint

    def getpid(self): return self.pid
    def getppid(self): return self.ppid
    def getuid(self): return self.uid
    def getgeneral(self): return self.general
    def getorderCount(self): return self.orderCount
    def getPSW(self): return self.PSW
    def getstackPoint(self): return self.stackPoint
    def getstats(self): return self.stats
    def getpriority(self): return self.priority
    def getother(self): return self.other
    def getincident(self): return self.incident
    def getlocation(self): return self.location
    def getsynCommunication(self): return self.synCommunication
    def getresource(self): return self.resource
    def getlistPoint(self): return self.listPoint
    def getAll(self):
        AllList = [self.pid,self.ppid,self.uid,self.general,
                   self.orderCount,self.PSW,self.stackPoint,
                   self.stats,self.priority,self.other,self.incident,
                   self.location,self.synCommunication,self.resource,self.listPoint]
        return AllList


class LPCB(): # 进程总链
    def __init__(self,x=None):
        self.data=x
        self.next=None


class MyQueue(): # 就绪队列
    def __init__(self,item=None):
        self.data = item
        self.next = None


def build_tree(root, btree, ppid):
    pre = None
    cur = root
    while cur:
        if cur.data.getpid() == ppid:
            if cur.son:
                cur = cur.son
                while cur:
                    pre = cur
                    cur = cur.brother
                pre.brother = btree
            else:
                cur.son = btree
        else:
            cur = cur.son


def main():
    head = LPCB()
    cur = head
    qHead, qEnd = None, None
    root = None # 树根
    pid_list = []
    num = int(input("请输入想创建的进程数："))
    for i in range(num):
        while 1:
            pid = int(input("请输入进程号pid："))
            if pid in pid_list:
                print("该进程已存在！请从新输入")
            else:
                pid_list.append(pid)
                break
        while 1:
            if i != 0:
                ppid = int(input("请输入该进程的父进程号pid："))
                if ppid not in pid_list:
                    print("该父进程不存在！请从新输入")
                else:
                    pid_list.append(ppid)
                    break
            else:
                ppid = os.getpid()
                break
        tmp = LPCB() # PCB总链
        Q = MyQueue() # 就绪队列
        tmp.data = PCB(pid,ppid,Uid,'进程%s通用寄存器'%i,random.randint(1001,2000),random.randint(2001,3000),
                       random.randint(3001,4000),'stop',random.randint(10,100),
                       random.randint(1,9),'wait',(random.randint(1,20),random.randint(0,15)),
                       random.randint(101,200),(random.randint(401,500),random.randint(501,600)),
                       random.randint(201,300))# 动态创建进程
        Q.data = tmp.data
        bTree = BTree(tmp.data)  # 进程树
        Q.next = None
        if qHead==None:
            qHead=qEnd=Q
        else:
            qEnd.next = Q
            qEnd = Q
        tmp.next = None
        cur.next = tmp
        cur = tmp
        time.sleep(0.1)
        print('进程%s创建成功'%(i+1))
        # 进程树
        if root is None:
            root = bTree
            continue
        build_tree(root, bTree, ppid)  # 创建进程树
    # es = qHead  # 查看进程队列
    # while es:
    #     print(es.data.getppid())
    #     es = es.next
    vue = head.next  # 查看进程LPCB
    while vue:
        print(vue.data.getAll())
        vue = vue.next


Uid = 1702
if __name__ == '__main__':
    main()



