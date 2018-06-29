class TreeNode:
    def __init__(self, key, value, left=None, right=None, parent=None):
        self.key = key
        self.value = value
        self.left_child = left
        self.right_child = right
        self.parent = parent

    def __repr__(self):
        params = {'key': self.key,
                  'value': self.value
                  }

        return '%s(%r)' % (self.__class__.__name__, params)

    def __iter__(self):
        if self:
            if self.left_child is not None:
                for elem in self.left_child:
                    yield elem
            yield self.key
            if self.right_child is not None:
                for elem in self.right_child:
                    yield elem

    def is_left_child(self):
        return self.parent is not None and self.parent.left_child == self

    def is_right_child(self):
        return self.parent is not None and self.parent.right_child == self

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return self.left_child is None and self.right_child is None

    def replace_data(self, key, value, lc=None, rc=None):
        self.key = key
        self.value = value
        self.left_child = lc
        self.right_child = rc

        if self.left_child is not None:
            self.left_child.parent = self
        if self.right_child is not None:
            self.right_child.parent = self

    def find_next_largest(self):
        """Find next-largest value"""
        successor = None
        if self.right_child is not None:
            successor = successor.right_child.find_min()

        else:
            if self.parent is not None:  # not root
                # parent would be next largest for a left child
                if self.is_left_child():
                    successor = self.parent
                else:
                    # if this node is right child and has no right child itself
                    # then its successor is the successor of its parent,
                    # excluding this node
                    self.right_child = None
                    successor = self.parent.find_next_largest()
                    self.right_child = self
        return successor

    def find_min(self):
        """Find the smallest value in the (sub)tree"""
        current = self
        while current.left_child is not None:
            current = current.left_child

        return current

    def splice_out(self):
        """Helper method for deleting a key from tree"""
        if self.is_leaf():
            if self.is_left_child():
                self.parent.left_child = None
            else:  # right child
                self.parent.right_child = None

        elif self.left_child is not None or \
                self.right_child is not None:
            # this is not general deletion
            # this only applies to "successor" nodes
            # which are guaranteed to have at most one child

            if self.left_child is not None:
                if self.is_left_child():
                    self.parent.left_child = self.left_child
                else:
                    self.parent.right_child = self.left_child
                self.left_child.parent = self.parent

            else:
                if self.is_left_child:
                    self.parent.left_child = self.right_child
                else:
                    self.parent.right_child = self.right_child
                self.right_child.parent = self.parent


class BinarySearchTree:
    def __init__(self):
        self.root = None
        self.size = 0

    def length(self):
        return self.size

    def __len__(self):
        return self.size

    def __iter__(self):
        return self.root.__iter__()

    def __setitem__(self, k, v):
        self.put(k, v)

    def __getitem__(self, k):
        return self.get(k)

    def __contains__(self, k):
        if self._get(k):
            return True

        return False

    def __delitem__(self, k):
        self.delete(k)

    def _get(self, key, current_node):
        if current_node is None:
            return None
        elif key == current_node.key:
            return current_node
        elif key < current_node.key:
            return self._get(key, current_node.left_child)
        elif key > current_node.key:
            return self._get(key, current_node.right_child)

    def _put(self, key, value, current_node):

        if key < current_node.key:
            if current_node.left_child is not None:
                self._put(key, value, current_node.left_child)
            else:
                current_node.left_child = TreeNode(
                    key, value, parent=current_node)

        elif key > current_node.key:
            if current_node.right_child is not None:
                self._put(key, value, current_node.right_child)
            else:
                current_node.right_child = TreeNode(
                    key, value, parent=current_node)

        else:
            current_node.key = key
            current_node.value = value

    def get(self, key):
        if self.root is not None:
            result = self._get(key, self.root)
            if result is not None:
                return result.value

        return None

    def put(self, key, value):

        if self.root is not None:
            self._put(key, value, self.root)
        else:
            self.root = TreeNode(key, value)

        self.size += 1

    def delete(self, key):

        if self._get(key, self.root) is None:
            raise KeyError('Key not in tree.')

        if self.size == 1 and self.current_node.key == key:
            self.root = None
            self.size -= 1

        else:  # size > 1
            node_to_remove = self._get(key, self.root)
            if node_to_remove is not None:
                self.remove(node_to_remove)

    def remove(self, node):
        if node.is_leaf():
            if node.parent.left_child == node:
                node.parent.left_child = None
            else:
                node.parent.right_child = None

        elif node.left_child is not None and \
                node.right_child is not None:
            successor = node.find_next_largest()
            node.splice_out()
            node.key = successor.key
            node.value = successor.value

        else:  # node has only one child
            # if only one child, promote it to take its place
            if node.left_child is not None:
                if node.is_left_child():
                    node.left_child.parent = node.parent
                    node.parent.left_child = node.left_child
                elif node.is_right_child():
                    node.left_child.parent = node.parent
                    node.parent.right_child = node.left_child
                else:  # is root
                    node.replace_data(node.left_child.key,
                                      node.left_child.value,
                                      node.left_child.left_child,
                                      node.left_child.right_child)

            else:  # only has right_child
                if node.is_left_child():
                    node.right_child.parent = node.parent
                    node.parent.left_child = node.right_child
                elif node.is_right_child():
                    node.right_child.parent = node.parent
                    node.parent.right_child = node.right_child
                else:  # is root
                    node.replace_data(node.right_child.key,
                                      node.right_child.value,
                                      node.right_child.left_child,
                                      node.right_child.right_child)
