import numpy as np
from Node import DictNode


class DictDiscriminator:
    def __init__(self, n_nodes: int, mapping: dict[int, np.ndarray]):
        self.mapping = mapping
        self.nodes = [DictNode() for _ in range(n_nodes)]

    def train(self, X_train: np.ndarray):
        for row in X_train:
            for i, node in enumerate(self.nodes):
                tup = "".join(row[self.mapping[i]])
                node.train(tup)

    def get_response(self, row: np.ndarray, bleaching: int = 1) -> int:
        indices = np.array([self.mapping[i] for i in range(len(self.nodes))])
        tuples = np.array(["".join(r) for r in row[indices]])
        return sum(
            node.get_response(tup, bleaching) for tup, node in zip(tuples, self.nodes)
        )
