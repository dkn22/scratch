class Queue:

    def __init__(self):
        self.items = []

    def isEmpty(self):
        return False if self.items else True

    def enqueue(self, item):
        self.items.insert(0, item)

    def size(self):
        return len(self.items)

    def dequeue(self):
        return self.items.pop()


def hot_potato(names, num):
    queue = Queue()

    for name in names:
        queue.enqueue(name)

    while queue.size() > 1:
        for i in range(num):
            # first person goes to back of the queue
            queue.enqueue(queue.dequeue)

        # every num-th person gets removed
        queue.dequeue()

    return queue.dequeue()


class Deque:

    def __init__(self):
        self.items = []

    def addFront(self, item):
        self.items.append(item)

    def addRear(self, item):
        self.items.insert(0, item)

    def size(self):
        return len(self.items)

    def isEmpty(self):
        return False if self.items else True

    def removeFront(self):
        return self.items.pop()

    def removeRear(self):
        return self.items.pop(0)


def is_palindrome(string):

    deque = Deque()
    for s in string:
        deque.addFront(s)

    ispalindrome = True

    while deque.size() > 1 and ispalindrome:
        last = deque.removeRear()
        first = deque.removeFront()

        if first != last:
            ispalindrome = False

    return ispalindrome
