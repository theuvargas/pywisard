import numpy as np
from collections.abc import Callable
import math


class BloomFilter:
    def __init__(
        self, size: int, num_hashes: int, dtype: type = bool, hash_fn: Callable = hash
    ):
        self.arr = np.zeros(size, dtype=dtype)
        self.num_hashes = num_hashes
        self.hash_fn = hash_fn
        self.dtype = dtype
        self.hash_shift = 4

    def add(self, item: tuple[np.int8]):
        h = self.hash_fn(item)
        h2 = h >> self.hash_shift
        for i in range(self.num_hashes):
            hash_val = (h + i * h2) % len(self.arr)
            if self.dtype == bool:
                self.arr[hash_val] = 1
            else:
                self.arr[hash_val] += 1

    def delete(self, item: tuple[np.int8]):
        if self.dtype == bool:
            raise ValueError("Cannot delete from a binary Bloom Filter")

        h = self.hash_fn(item)
        h2 = h >> self.hash_shift
        for i in range(self.num_hashes):
            hash_val = (h + i * h2) % len(self.arr)
            if self.arr[hash_val] == 0:
                return
            self.arr[hash_val] -= 1

    def min_membership(self, item: tuple[np.int8]) -> int:
        h = self.hash_fn(item)
        h2 = h >> self.hash_shift
        vals = []
        for i in range(self.num_hashes):
            hash_val = (h + i * h2) % len(self.arr)
            vals.append(self.arr[hash_val])
        return min(vals)


def bloom_filter_parameters(n, p) -> tuple[int, int]:
    m = -(n * math.log(p)) / (math.log(2) ** 2)
    k = (m / n) * math.log(2)

    m = math.ceil(m)
    k = round(k)

    return m, k
