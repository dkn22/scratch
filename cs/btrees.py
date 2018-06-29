class BinaryTree(object):
    """
    List of lists implementation of binary trees.
    """

    def __init__(self, root):
        self.tree = [root, [], []]

    @classmethod
    def insert_left(cls, root, branch):
        t = root.pop(1)
        if len(t) > 1:
            # if node already exists, push it down
            root.insert(1, [branch, t, []])
        else:
            root.insert(1, [branch, [], []])

        return root

    @classmethod
    def insert_right(cls, root, branch):
        t = root.pop(2)
        if len(t) > 1:
            # if node already exists, push it down
            root.insert(2, [branch, [], t])
        else:
            root.insert(2, [branch, [], []])

        return root

    def get_root(self, root=None):
        if root is None:
            root = self.tree
        return root[0]

    def set_root(self, value, root=None):
        if root is None:
            root = self.tree

        root[0] = value

        return root

    def left_child(self, root=None):
        if root is None:
            root = self.tree

        return root[1]

    def right_child(self, root=None):
        if root is None:
            root = self.tree
            
        return root[2]