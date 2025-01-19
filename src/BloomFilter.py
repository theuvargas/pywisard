import numpy as np
from collections.abc import Callable


class BloomFilter:
    def __init__(
        self, size: int, num_hashes: int, dtype: type = bool, hash_fn: Callable = hash
    ):
        self.arr = np.zeros(size, dtype=dtype)
        self.num_hashes = num_hashes
        self.hash_fn = hash_fn
        self.dtype = dtype

    def add(self, item: str):
        for i in range(self.num_hashes):
            hash_val = self.hash_fn(item + str(i)) % len(self.arr)
            if self.dtype == bool:
                self.arr[hash_val] = 1
            else:
                self.arr[hash_val] += 1

    def delete(self, item: str):
        if self.dtype == bool:
            raise ValueError("Cannot delete from a binary Bloom Filter")

        for i in range(self.num_hashes):
            hash_val = self.hash_fn(item + str(i)) % len(self.arr)
            if self.arr[hash_val] == 0:
                return
            self.arr[hash_val] -= 1

    def min_membership(self, item: str) -> int:
        vals = []
        for i in range(self.num_hashes):
            hash_val = self.hash_fn(item + str(i)) % len(self.arr)
            vals.append(self.arr[hash_val])
        return min(vals)
