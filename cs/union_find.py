class UnionFind:
    def __init__(self, ids):
        self._id = {i: i for i in ids}
        self._sz = {i: 1 for i in ids}
        self.n_components = len(self._sz)

    def _root(self, i):
        j = i
        while (j != self._id[j]):
            self._id[j] = self._id[self._id[j]]
            j = self._id[j]
        return j

    def find(self, p):
        return self._root(p)

    def union(self, p, q):
        i = self._root(p)
        j = self._root(q)

        # union by size
        if (self._sz[i] < self._sz[j]):
            self._id[i] = j
            self._sz[j] = self._sz[j] + self._sz.pop(i)
        else:
            self._id[j] = i
            self._sz[i] = self._sz[i] + self._sz.pop(j)
            
        self.n_components -= 1