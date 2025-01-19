import numpy as np
from Node import DictNode


class DictDiscriminator:
    def __init__(self, n_nodes: int, mapping: dict[int, np.ndarray]):
        self.mapping = mapping
        self.nodes = [DictNode() for _ in range(n_nodes)]

    def train(self, X_train: np.ndarray):
        for row in X_train:
            x = "".join(row.astype(str))
            for i, node in enumerate(self.nodes):
                node.train(x[self.mapping[i]])
