import random


def product_random():  # 生成随机页面
    queue = []
    while 1:
        random_ = random.randint(0, 319)
        last = random_
        random_ = (random_ + 1) // 10
        queue.append(random_)
        if len(queue) == 320:
            break
        random_ = random.randint(0, last)
        start = random_ + 2
        random_ = (random_ + 1) // 10
        queue.append(random_)
        if len(queue) == 320:
            break
        random_ = random.randint(start, 319)
        random_ = random_ // 10
        queue.append(random_)
    return queue


def check_func(number, physical_block, full):
    if not physical_block:
        return False
    for i in range(0, full):
        if number == physical_block[i]:  # 存在返回真
            return True
    return False


def find_best(check, flag, queue, physical_block):  # 找出最长时间不被访问的页号
    lists = [0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(0, 8):  # 八个数
        for j in range(check, 320):  # 逐个寻找到下一个相同页号的距离
            if physical_block[i] == queue[j]:
                lists[i] = j - flag
                break
    if lists:  # 寻找最大数的下标
        max_ = lists[0]
        num = 0
        for i in range(1, 8):
            if lists[i] == 0:
                return i
            if max_ < lists[i]:
                max_ = lists[i]
                num = i
        return num


def opt(queue):
    physical_block = []  # 物理块
    counts, full = 0, 0  # 缺页数量  物理块计数
    for flag in range(0, 320):
        if full < 8:  # 数组内数字数量小于8时
            if not check_func(queue[flag], physical_block, full):   # 查重
                physical_block.append(queue[flag])
                counts += 1
                full += 1
            flag += 1
        else:
            if not check_func(queue[flag], physical_block, full):
                check = flag + 1  # 下一个访问页号
                num = find_best(check, flag, queue, physical_block)  # 找出最长时间不被访问的页号
                physical_block[num] = queue[flag]
                counts += 1
            flag += 1
    print("OPT缺页率为：", counts / 320.0)


def fifo(queue):
    physical_block = []
    counts, full = 0, 0
    for flag in queue:
        if full < 8:
            if not check_func(flag, physical_block, full):
                physical_block.append(flag)
                counts += 1
                full += 1
        else:
            if not check_func(flag, physical_block, full):
                physical_block.pop(0)  # 弹出最先到的页
                physical_block.append(flag)  # 将新页加入内存
                counts += 1
    print("FIFO缺页率为：", counts / 320.0)


def exchange(flag, physical_block, full):
    for i in range(0, full):
        if physical_block[i] == flag:
            return i


def lru(queue):
    physical_block = []
    counts, full = 0, 0
    for flag in queue:
        if full < 8:
            if not check_func(flag, physical_block, full):
                physical_block.append(flag)
                counts += 1
                full += 1
            else:
                index = exchange(flag, physical_block, full)  # 最近访问的页面后移
                physical_block.pop(index)
                physical_block.append(flag)
        else:
            if not check_func(flag, physical_block, full):
                physical_block.pop(0)
                physical_block.append(flag)
                counts += 1
            else:
                index = exchange(flag, physical_block, full)  # 最近访问的页面后移
                physical_block.pop(index)
                physical_block.append(flag)
    print("LRU缺页率为：", counts / 320.0)


def main():
    queue = product_random()  # 缓存列表
    opt(queue)  # 最佳置换算法
    fifo(queue)  # 先进先出
    lru(queue)  # 最近最久未使用


if __name__ == '__main__':
    main()
