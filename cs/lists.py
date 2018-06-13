class Node:

    def __init__(self, data):
        self.data = data
        self.next = None

    def getData(self):
        return self.data

    def getNext(self):
        return self.next

    def setData(self, data):
        self.data = data

    def setNext(self, new):
        self.next = new


class UnorderedList:

    def __init__(self):
        self.head = None

    def isEmpty(self):
        return self.head is None

    def add(self, item):
        new = Node(item)
        # next node is current head
        new.setNext(self.head)
        # set current head to be this new item
        self.head = new

    def size(self):
        count = 0
        current_node = self.head
        while current_node is not None:  # linked list traversal
            count += 1
            current_node = current_node.getNext()

        return count

    def search(self, item):
        current_node = self.head
        found = False

        while current_node is not None and not found:
            if current_node.getData() == item:
                found = True
            else:
                current_node = current_node.getNext()

        return found

    def remove(self, item):
        current_node = self.head
        previous_node = None
        found = False

        while current_node is not None and not found:
            if current_node.getData() == item:
                found = True
                # previous_node.setNext(current_node.getNext())

            else:
                previous_node = current_node
                current_node = current_node.getNext()

        if current_node is None:
            raise ValueError('Item not found.')

        if previous_node is None:  # head contains the item
            self.head = current_node.getNext()
        else:  # item is not in the head
            previous_node.setNext(current_node.getNext())

    def append(self, item):
        new_node = Node(item)
        current_node = self.head

        while current_node.getNext() is not None:
            current_node = current_node.getNext()

        current_node.setNext(new_node)

    def pop(self, index=None):

        current_node = self.head
        previous_node = None

        if index is None:
            while current_node.getNext() is not None:
                current_node = current_node.getNext()

            current_node.setNext(None)

        elif index == 0:
            self.head = current_node.getNext()

        else:
            for i in range(index):
                previous_node = current_node
                current_node = current_node.getNext()

            previous_node.setNext(current_node.getNext())

    def index(self, item):
        if not self.search(item):
            raise ValueError('Item not present.')

        idx = 0
        current_node = self.head
        found = False

        while current_node is not None and not found:
            if current_node.getData() == item:
                found = True
            else:
                idx += 1
                current_node = current_node.getNext()

        return idx


class OrderedList(UnorderedList):

    def __init__(self):
        super(OrderedList, self).__init__()

    def search(self, item):
        current_node = self.head
        found = False

        while current_node is not None and not found \
                and item >= current_node.getData():
            if current_node.getData() == item:
                found = True
            else:
                current_node = current_node.getNext()

        return found

    def add(self, item):
        current_node = self.head
        previous_node = None

        while current_node is not None and current_node.getData() <= item:
            previous_node = current_node
            current_node = current_node.getNext()

        new = Node(item)

        if previous_node is None:
            new.setNext(self.head)
            self.head = new

        else:
            new.setNext(current_node)
            previous_node.setNext(new)
