from collections import namedtuple
# A Binary Tree Node


class Node:

    def __init__(self, data):
        self.data = data

        # left child
        self.left = None

        # right child
        self.right = None


def max_depth(root):
    if root is None:
        return -1

    left_d = max_depth(root.left)
    right_d = max_depth(root.right)

    return max(left_d, right_d) + 1


def max_span(root):
    """Max span and depth of a binary tree"""

    Properties = namedtuple('Properties', ['max_span', 'max_depth'])

    if root is None:
        # first element is span, second is depth
        return Properties(-1, -1)

    left = max_span(root.left)
    right = max_span(root.right)

    span = max(2 + left[1] + right[1], left[0], right[0])
    depth = max(left[1], right[1]) + 1

    return Properties(span, depth)
