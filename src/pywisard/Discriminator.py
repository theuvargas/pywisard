import numpy as np
from pywisard.Node import BloomNode, DictNode


class DictDiscriminator:
    def __init__(
        self,
        n_nodes: int,
        mapping: dict[int, np.ndarray],
        dtype: type,
        withBleaching: bool,
    ):
        self.mapping = mapping
        self.nodes = [DictNode(dtype, withBleaching) for _ in range(n_nodes)]

    def train(self, X_train: np.ndarray):
        for row in X_train:
            for i, node in enumerate(self.nodes):
                tup = tuple(row[self.mapping[i]])
                node.train(tup)

    def get_response(self, row: np.ndarray, bleaching: int = 1) -> int:
        indices = np.array([self.mapping[i] for i in range(len(self.nodes))])
        tuples = [tuple(r) for r in row[indices]]
        return sum(
            node.get_response(tup, bleaching) for tup, node in zip(tuples, self.nodes)
        )


class BloomDiscriminator:
    def __init__(
        self,
        n_nodes: int,
        mapping: dict[int, np.ndarray],
        bloom_size: int,
        num_hashes: int,
        dtype: type,
        withBleaching: bool,
    ) -> None:
        self.mapping = mapping
        self.nodes = [
            BloomNode(bloom_size, num_hashes, dtype, withBleaching)
            for _ in range(n_nodes)
        ]

    def train(self, X_train: np.ndarray) -> None:
        for row in X_train:
            for i, node in enumerate(self.nodes):
                tup = tuple(row[self.mapping[i]])
                node.train(tup)

    def get_response(self, row: np.ndarray, bleaching: int = 1) -> int:
        indices = np.array([self.mapping[i] for i in range(len(self.nodes))])
        tuples = [tuple(r) for r in row[indices]]
        return sum(
            node.get_response(tup, bleaching) for tup, node in zip(tuples, self.nodes)
        )
