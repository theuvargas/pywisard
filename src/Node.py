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

    @abstractmethod
    def untrain(self, input: str) -> None:
        return None


class DictNode(Node):
    def __init__(self):
        self.memory = {}

    def get_response(self, input: str) -> int:
        return self.memory.get(input, 0)

    def train(self, input: str):
        if input in self.memory:
            self.memory[input] += 1
        else:
            self.memory[input] = 1

    def untrain(self, input: str) -> None:
        if input not in self.memory or self.memory[input] == 0:
            raise KeyError(f"Cannot untrain '{input}' as it is not in memory")

        self.memory[input] -= 1

    def __str__(self):
        return str(self.memory)


class BloomNode(Node):
    def __init__(
        self,
        bloom_size: int,
        num_hashes: int,
        dtype: type = bool,
        hash_fn: Callable = hash,
    ):
        self.memory = BloomFilter(bloom_size, num_hashes, dtype, hash_fn)

    def get_response(self, input: str) -> int:
        return int(self.memory.is_member(input))

    def train(self, input: str):
        self.memory.add(input)

    def untrain(self, input: str) -> None:
        if input not in self.memory:
            raise KeyError(f"Cannot untrain '{input}' as it is not in memory")

        self.memory.delete(input)

    def __str__(self):
        return str(self.memory.arr)
