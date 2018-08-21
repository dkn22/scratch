class MinBinHeap:
    def __init__(self):
        self.heaplist = [0]
        self.current_size = 0

    def __repr__(self):
        params = {'heaplist': self.heaplist,
                  'current_size': self.current_size
                  }

        return '%s(%r)' % (self.__class__.__name__, params)

    def perc_up(self, i):
        while i // 2 > 0: # while has 'parent'
            if self.heaplist[i] < self.heaplist[i // 2]:
                self.heaplist[i // 2], self.heaplist[i] =\
                    self.heaplist[i], self.heaplist[i // 2]

            i = i // 2

    def insert(self, value):
        self.heaplist.append(value)
        self.current_size += 1
        self.perc_up(self.current_size)

    def find_min(self):
        return self.heaplist[1]

    def min_child(self, i):
        if i * 2 + 1 > self.current_size:
            return i * 2

        else:
            if self.heaplist[i * 2] < self.heaplist[i * 2 + 1]:
                return i * 2
            else:
                return i * 2 + 1

    def perc_down(self, i):
        while (i * 2) <= self.current_size:
            mc = self.min_child(i)
            if self.heaplist[i] > self.heaplist[mc]:
                self.heaplist[i], self.heaplist[mc] =\
                    self.heaplist[mc], self.heaplist[i]

            i = mc

    def del_min(self):
        root = self.find_min()

        self.heaplist[1] = self.heaplist[self.current_size]
        self.current_size -= 1
        self.heaplist.pop()
        self.perc_down(1)

        return root

    @classmethod
    def build_heap(cls, alist):
        heap = MinBinHeap()
        heap.heaplist = [0] + alist[:]
        heap.current_size = len(alist)
        i = heap.current_size // 2

        while i > 0:
            heap.perc_down(i)
            i -= 1

        return heap


class PriorityQueue:
    def __init__(self):
        self.heaplist = [(0, 0)]
        self.current_size = 0

    def build_heap(self, alist):
        self.current_size = len(alist)
        self.heaplist = [(0, 0)]

        for i in alist:
            self.heaplist.append(i)

        i = len(alist) // 2 
        while i > 0:
            self.perc_down(i)
            i = i - 1

    def perc_down(self, i):
        while i * 2 <= self.current_size:
            mc = self.min_child(i)
            if self.heaplist[i][0] > self.heaplist[mc][0]:
                self.heaplist[i], self.heaplist[mc] =\
                    self.heaplist[mc], self.heaplist[i]
            i = mc

    def min_child(self, i):
        if i * 2 > self.current_size:
            return -1
        else:
            if i * 2 + 1 > self.current_size:
                return i * 2
            else:
                if self.heaplist[i * 2][0] < self.heaplist[i * 2 + 1][0]:
                    return i * 2
                else:
                    return i * 2 + 1

    def perc_up(self, i):
        while i // 2 > 0:
            if self.heaplist[i][0] < self.heaplist[i // 2][0]:
                self.heaplist[i], self.heaplist[i // 2] =\
                    self.heaplist[i // 2], self.heaplist[i]
            i = i // 2

    def insert(self, item):
        self.heaplist.append(item)
        self.current_size += 1
        self.perc_up(self.current_size)

    def del_min(self):
        min_item = self.heaplist[1][1]

        self.heaplist[1] = self.heaplist[self.current_size]
        self.current_size -= 1
        self.heaplist.pop()
        self.perc_down(1)

        return min_item

    def is_empty(self):
        return self.current_size == 0

    def decrease_key(self, val, amount):
        item = [tup for tup in self.heaplist if tup[1] == val]

        if item:
            key = self.heaplist.index(item[0])

        if key > 0:
            self.heaplist[key] = (amount, self.heaplist[key][1])
            self.perc_up(key)

    def __contains__(self, val):
        return len([tup for tup in self.heaplist if tup[1] == val]) > 0


class MaxBinHeap:
    def __init__(self):
        self.heaplist = [0]
        self.current_size = 0

    def __repr__(self):
        params = {'heaplist': self.heaplist,
                  'current_size': self.current_size
                  }

        return '%s(%r)' % (self.__class__.__name__, params)

    def perc_up(self, i):
        while i // 2 > 0: # while has 'parent'
            if self.heaplist[i] > self.heaplist[i // 2]:
                self.heaplist[i // 2], self.heaplist[i] =\
                    self.heaplist[i], self.heaplist[i // 2]

            i = i // 2

    def insert(self, value):
        self.heaplist.append(value)
        self.current_size += 1
        self.perc_up(self.current_size)

    def find_max(self):
        return self.heaplist[1]

    def max_child(self, i):
        if i * 2 + 1 > self.current_size:
            return i * 2

        else:
            if self.heaplist[i * 2] > self.heaplist[i * 2 + 1]:
                return i * 2
            else:
                return i * 2 + 1

    def perc_down(self, i):
        while (i * 2) <= self.current_size:
            mc = self.max_child(i)
            if self.heaplist[i] < self.heaplist[mc]:
                self.heaplist[i], self.heaplist[mc] =\
                    self.heaplist[mc], self.heaplist[i]

            i = mc

    def del_max(self):
        root = self.find_max()

        self.heaplist[1] = self.heaplist[self.current_size]
        self.current_size -= 1
        self.heaplist.pop()
        self.perc_down(1)

        return root

    @classmethod
    def build_heap(cls, alist):
        heap = MaxBinHeap()
        heap.heaplist = [0] + alist[:]
        heap.current_size = len(alist)
        i = heap.current_size // 2

        while i > 0:
            heap.perc_down(i)
            i -= 1

        return heap
