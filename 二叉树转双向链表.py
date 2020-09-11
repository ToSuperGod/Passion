class bTree(object):
    def __init__(self):
        self.data = None
        self.left = None
        self.right = None


class exchange(object):
    def __init__(self):
        self.pHead = None
        self.pEnd = None

    def build_tree(self, array, start, end):  # 递归建立二叉树
        root = bTree()
        if end >= start:
            mid = (start + end + 1) // 2
            root.data = array[mid]
            root.left = self.build_tree(array, start, mid - 1)
            root.right = self.build_tree(array, mid + 1, end)
        else:
            root = None
        return root

    def reform(self, root):  # 开始转换
        if root is None:
            return
        self.reform(root.left)
        root.left = self.pEnd
        if self.pEnd is None:
            self.pEnd = self.pHead = root
        else:
            self.pEnd.right = root
        self.pEnd = root
        self.reform(root.right)


def main():
    arr = [1, 2, 3, 4, 5, 6, 7]
    user = exchange()
    root = user.build_tree(arr, 0, len(arr) - 1)
    user.reform(root)
    cur = user.pHead
    print("双向链表顺序遍历")
    while cur:
        print(cur.data, end='')
        cur = cur.right
    pre = user.pEnd
    print()
    print("双向链表逆向遍历")
    while pre:
        print(pre.data, end='')
        pre = pre.left


if __name__ == '__main__':
    main()
