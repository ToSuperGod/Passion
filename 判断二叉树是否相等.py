class bTree(object):
    def __init__(self):
        self.data = None
        self.left = None
        self.right = None


def is_equal(root1, root2):
    if root1 is None and root2 is None:
        return True
    if root2 is None and root1 is not None:
        return False
    if root1 is None and root2 is not None:
        return False
    if root1.data == root2.data:
        return is_equal(root1.left, root2.left) and is_equal(root1.right, root2.right)
    else:
        return False


def build_tree():
    root = bTree()
    node1 = bTree()
    node2 = bTree()
    node3 = bTree()
    node4 = bTree()
    root.data = 6
    node1.data = 3
    node2.data = 8
    node3.data = 4
    node4.data = 7
    root.left = node1
    root.right = node2
    node1.left = node3
    node1.right = node4
    node2.right = node2.left = node3.left = node3.right = node4.left = node4.right = None
    return root


def main():
    root1 = build_tree()
    root2 = build_tree()
    if is_equal(root1, root2):
        print("相等")
    else:
        print("不等")
    pass


if __name__ == '__main__':
    main()
