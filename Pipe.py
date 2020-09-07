from multiprocessing import Process,Pipe,Lock
import os
import time

def workOne(right,mutex):
    mutex.acquire()
    right.send("Message from Child1")
    mutex.release()

def workTwo(right,mutex):
    mutex.acquire()
    right.send("Message from Child2")
    mutex.release()

if __name__ == '__main__':
    left,right = Pipe()
    mutex = Lock()
    p1 = Process(target=workOne,args=(right,mutex))
    p2 = Process(target=workTwo,args=(right,mutex))
    p1.start()
    p2.start()
    print(left.recv())
    print(left.recv())
    left.close()
    right.close()
