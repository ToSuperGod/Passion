from collections import deque
import random
import csv
import time


class PCB(object):  # 进程块
    def __init__(self, pid=None, priority=None, need_time=None, used_time=0, time_split=0, state='未完成'):
        self.pid = pid  # 唯一ID
        self.priority = priority  # 优先级
        self.need_time = need_time  # 需要时间
        self.used_time = used_time  # 已执行时间
        self.time_split = time_split  # 执行时间片
        self.state = state  # 状态
        # self.next = None # 指针

    def get_pid(self): return self.pid
    def get_priority(self): return self.priority
    def get_need_time(self): return self.need_time
    def get_used_time(self): return self.used_time
    def get_time_split(self): return self.get_time_split
    def get_state(self): return self.state


def operation_os(list_queue):
    path = './operation.csv'
    process_file = open(path, 'a+', newline='', encoding='gbk')
    writer = csv.writer(process_file)
    writer.writerow(('pid', 'priority', 'need_time', 'used_time', 'time_split', 'state'))  # csv文件首行
    for queue in list_queue:
        while queue:
            process = queue.popleft()  # 队列中左侧弹出进程
            print("正在执行进程%d" % (process.get_pid()))
            execute_time = random.randint(1, 5)  # 执行时间片
            process.time_split = execute_time  # 赋值
            process.used_time += execute_time  # 已执行时间
            time_ = 0
            if process.used_time >= process.need_time:  # 判断是否执行完毕
                time_ = process.used_time - process.need_time  # 超过时间
                process.state = "执行完毕"
                print("进程%d执行完毕" % (process.get_pid()))
            else:
                queue.append(process)  # 未执行完毕加入队列结尾
            time.sleep((execute_time-time_)/10)  # 模拟延时
            used = process.get_used_time() - time_
            writer.writerow((process.get_pid(), process.get_priority(),  # 将执行状态写入文件
                             process.get_need_time(), used,
                             execute_time, process.get_state()))
    print("所有进程执行完毕！！！")
    process_file.close()


def create_process():
    queue_0 = deque()
    queue_1 = deque()
    queue_2 = deque()
    queue_3 = deque()
    queue_4 = deque()
    queue_5 = deque()
    queue_6 = deque()
    queue_7 = deque()
    queue_8 = deque()
    path = './init.csv'
    process_file = open(path, 'a+', newline='', encoding='gbk')  # 创建CSV文件
    writer = csv.writer(process_file)
    writer.writerow(('pid', 'priority', 'need_time', 'used_time', 'time_split', 'state'))  # csv文件首行
    try:
        while 1:
            nums = int(input("请输入要执行的进程数(n>20)："))
            if nums > 20:
                break
            else:
                print("注意进程数n>20！！")
        for i in range(1, nums+1):
            pid = i
            priority = random.randint(0, 8)
            need_time = random.randint(1, 10)
            process = PCB(pid, priority, need_time)
            if process.get_priority() == 0:
                queue_0.append(process)
            elif process.get_priority() == 1:
                queue_1.append(process)
            elif process.get_priority() == 2:
                queue_2.append(process)
            elif process.get_priority() == 3:
                queue_3.append(process)
            elif process.get_priority() == 4:
                queue_4.append(process)
            elif process.get_priority() == 5:
                queue_5.append(process)
            elif process.get_priority() == 6:
                queue_6.append(process)
            elif process.get_priority() == 7:
                queue_7.append(process)
            elif process.get_priority() == 8:
                queue_8.append(process)
            writer.writerow((pid, priority, need_time, process.get_used_time(),
                             process.get_time_split(), process.get_state()))
    except Exception as e:
        print("写入CSV文件错误")
    finally:
        print("进程创建完毕，并成功写入！")
        process_file.close()
    list_queue = [queue_0, queue_1, queue_2, queue_3, queue_4, queue_5, queue_6, queue_7, queue_8]
    operation_os(list_queue)  # 执行进程


def main():
    create_process()  # 创建进程


if __name__ == '__main__':
    main()

