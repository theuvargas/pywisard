from abc import ABC, abstractmethod
from BloomFilter import BloomFilter
from collections.abc import Callable


class Node(ABC):
    @abstractmethod
    def get_response(self, input: str) -> int:
        return 0

    @abstractmethod
    def train(self, input: str) -> None:
        return None


class DictNode(Node):
    def __init__(self):
        self.dictionary = {}

    def get_response(self, input: str) -> int:
        return self.dictionary.get(input, 0)

    def train(self, input: str):
        if input in self.dictionary:
            self.dictionary[input] += 1
        else:
            self.dictionary[input] = 1

    def __str__(self):
        return str(self.dictionary)


class BloomNode(Node):
    def __init__(
        self,
        bloom_size: int,
        num_hashes: int,
        dtype: type = bool,
        hash_fn: Callable = hash,
    ):
        self.bloom_filter = BloomFilter(bloom_size, num_hashes, dtype, hash_fn)

    def get_response(self, input: str) -> int:
        return int(self.bloom_filter.is_member(input))

    def train(self, input: str):
        self.bloom_filter.add(input)

    def __str__(self):
        return str(self.bloom_filter.arr)
