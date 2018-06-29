from binary_search_tree import BinarySearchTree, TreeNode


class AVLTree(BinarySearchTree):

    def __init__(self):
        super(AVLTree, self).__init__()

    def _put(self, key, value, current_node):

        if key < current_node.key:
            if current_node.left_child is not None:
                self._put(key, value, current_node.left_child)
            else:
                current_node.left_child = TreeNode(
                    key, value, parent=current_node)

                self.update_balance(current_node.left_child)

        elif key > current_node.key:
            if current_node.right_child is not None:
                self._put(key, value, current_node.right_child)
            else:
                current_node.right_child = TreeNode(
                    key, value, parent=current_node)

                self.update_balance(current_node.right_child)

    def update_balance(self, node):
        if node.balance_factor > 1 or node.balance_factor < -1:
            self.rebalance(node)
            return

        if node.parent is not None:
            if node.is_left_child():
                node.parent.balance_factor += 1

            elif node.is_right_child():
                node.parent.balance_factor -= 1

            if node.parent.balance_factor != 0:
                self.update_balance(node.parent)

    def rotate_left(self, old_root):
        new_root = old_root.right_child
        old_root.right_child = new_root.left_child

        if new_root.left_child is not None:
            new_root.left_child.parent = old_root
        new_root.parent = old_root.parent

        if old_root.is_root():
            self.root = new_root

        else:
            if old_root.is_left_child():
                old_root.parent.left_child = new_root
            else:
                old_root.parent.right_child = new_root

        new_root.left_child = old_root
        old_root.parent = new_root

        old_root.balance_factor = old_root.balance_factor + \
            1 - min(new_root.balance_factor, 0)
        new_root.balance_factor = new_root.balance_factor + \
            1 + max(old_root.balance_factor, 0)

    def rotate_right(self, old_root):
        new_root = old_root.left_child
        old_root.left_child = new_root.right_child

        if new_root.right_child is not None:
            new_root.right_child.parent = old_root
        new_root.parent = old_root.parent

        if old_root.is_root():
            self.root = new_root
        else:
            if old_root.is_left_child():
                old_root.parent.left_child = new_root
            else:
                old_root.parent.right_child = new_root

        new_root.right_child = old_root
        old_root.parent = new_root

        old_root.balance_factor = old_root.balance_factor - \
            1 - max(0, new_root.balance_factor)
        new_root.balance_factor = new_root.balance_factor - \
            1 + min(0, old_root.balance_factor)

    def rebalance(self, node):

        if node.balance_factor < 0:  # right-heavy
            if node.right_child.balance_factor > 0:
                # if child is left-heavy, rotate it right first
                self.rotate_right(node.right_child)
            self.rotate_left(node)

        elif node.balance_factor > 0:  # left-heavy
            if node.left_child.balance_factor < 0:
                # if child is right-heavy, rotate it left first
                self.rotate_left(node.left_child)
            self.rotate_right(node)

    def remove(self, node):
        if node.is_leaf():
            if node.parent.left_child == node:
                node.parent.left_child = None
                # node.parent.balance_factor -= 1

            else:
                node.parent.right_child = None
                # node.parent.balance_factor += 1

            self.update_balance_del(node.parent)

        elif node.left_child is not None and \
                node.right_child is not None:
            successor = node.find_next_largest()
            successor.splice_out()
            node.key = successor.key
            node.value = successor.value

            self.update_balance_del(successor)

        else:  # node has only one child
            # if only one child, promote it to take its place
            if node.left_child is not None:
                if node.is_left_child():
                    node.left_child.parent = node.parent
                    node.parent.left_child = node.left_child

                    # node.parent.balance_factor -= 1
                elif node.is_right_child():
                    node.left_child.parent = node.parent
                    node.parent.right_child = node.left_child

                    # node.parent.balance_factor += 1
                else:  # is root
                    node.replace_data(node.left_child.key,
                                      node.left_child.value,
                                      node.left_child.left_child,
                                      node.left_child.right_child)

            else:  # only has right_child
                if node.is_left_child():
                    node.right_child.parent = node.parent
                    node.parent.left_child = node.right_child

                    # node.parent.balance_factor -= 1
                elif node.is_right_child():
                    node.right_child.parent = node.parent
                    node.parent.right_child = node.right_child

                    # node.parent.balance_factor += 1
                else:  # is root
                    node.replace_data(node.right_child.key,
                                      node.right_child.value,
                                      node.right_child.left_child,
                                      node.right_child.right_child)

            # self.rebalance(node.parent)
            self.update_balance_del(node.parent)

    def update_balance_del(self, node):
        if node.balance_factor > 1 or node.balance_factor < -1:
            self.rebalance(node)
            return

        if node.parent is not None:
            if node.is_left_child():
                node.parent.balance_factor -= 1

            elif node.is_right_child():
                node.parent.balance_factor += 1

            if node.parent.balance_factor != 0:
                self.update_balance_del(node.parent)
