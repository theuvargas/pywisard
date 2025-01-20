import numpy as np
from Node import DictNode


class DictDiscriminator:
    def __init__(self, n_nodes: int, mapping: dict[int, np.ndarray]):
        self.mapping = mapping
        self.nodes = [DictNode() for _ in range(n_nodes)]

    def train(self, X_train: np.ndarray):
        for row in X_train:
            for i, node in enumerate(self.nodes):
                # tup = "".join(row[j] for j in self.mapping[i])
                tup = hash(tuple(row[j] for j in self.mapping[i]))
                node.train(tup)

    def get_response(self, row: np.ndarray, threshold: int = 1) -> int:
        response = 0
        for i, node in enumerate(self.nodes):
            # tup = "".join(row_str[j] for j in self.mapping[i])
            tup = hash(tuple(row[j] for j in self.mapping[i]))
            response += node.get_response(tup, threshold)
        return response
