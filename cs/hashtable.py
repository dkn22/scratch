class HashTable:

    def __init__(self, size=11):
        self.size = size
        self.slots = [None] * self.size
        self.data = [None] * self.size

    def hash(self, key):
        return key % self.size

    def rehash(self, old_hash, key):
        return (old_hash + 1) % self.size

    def put(self, key, item):

        hash_value = self.hash(key)

        if self.slots[hash_value] is None:
            self.slots[hash_value] = key
            self.data[hash_value] = item

        else:

            if self.slots[hash_value] == key:
                # if same key, replace data
                self.data[hash_value] = item

            else:  # collision
                next_hash = self.rehash(hash_value, key)

                while self.slots[next_hash] is not None and \
                        self.slots[next_hash] != key:
                    next_hash = self.rehash(next_hash, key)

                if self.slots[next_hash] is None:
                    self.slots[next_hash] = key
                    self.data[next_hash] = item

                elif self.slots[next_hash] == key:
                    self.data[next_hash] = item

                else:
                    raise TypeError('No more available slots.')

    def get(self, key, default=None):

        hash_value = self.hash(key)
        idx = hash_value

        while self.slots[idx] is not None:

            if self.slots[idx] == key:
                return self.data[idx]

            else:
                idx = self.rehash(idx, key)

                if idx == hash_value:  # returned to initial position
                    return default  # key not present

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, item):
        self.put(key, item)
