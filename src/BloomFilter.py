import numpy as np


class BloomFilter:
    def __init__(self, size: int, num_hashes: int, dtype: type = bool, hash_fn=hash):
        self.arr = np.zeros(size, dtype=dtype)
        self.num_hashes = num_hashes
        self.hash_fn = hash_fn

    def add(self, item: str):
        for i in range(self.num_hashes):
            hash_val = self.hash_fn(item + str(i)) % len(self.arr)
            self.arr[hash_val] += 1

    def is_member(self, item: str) -> bool:
        for i in range(self.num_hashes):
            hash_val = self.hash_fn(item + str(i)) % len(self.arr)
            if not self.arr[hash_val]:
                return False
        return True
