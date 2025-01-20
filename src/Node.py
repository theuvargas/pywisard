from abc import ABC, abstractmethod
from BloomFilter import BloomFilter
from collections.abc import Callable


class Node(ABC):
    @abstractmethod
    def get_response(self, input: int) -> int:
        return 0

    @abstractmethod
    def train(self, input: int) -> None:
        return None

    @abstractmethod
    def untrain(self, input: str) -> None:
        return None


class DictNode(Node):
    def __init__(self, dtype: type = int):
        self.memory = {}
        self.dtype = dtype

    def get_response(self, input: int, threshold: int = 1) -> int:
        value = self.memory.get(input, 0)
        return 1 if value >= threshold else 0

    def train(self, input: int):
        if input in self.memory and self.dtype != bool:
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

    def get_response(self, input: str, threshold: int = 1) -> int:
        min_val = self.memory.min_membership(input)

        return 1 if min_val >= threshold else 0

    def train(self, input: str):
        self.memory.add(input)

    def untrain(self, input: str) -> None:
        self.memory.delete(input)

    def __str__(self):
        return str(self.memory.arr)
