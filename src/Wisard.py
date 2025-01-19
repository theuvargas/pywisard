import numpy as np
from Discriminator import DictDiscriminator
from utils import binarize_dataset


class DictWisard:
    def __init__(self, n_tuples: int, n_nodes: int, bits_per_input: int):
        self.n_tuples = n_tuples
        self.n_nodes = n_nodes
        self.bits_per_input = bits_per_input
        self.discriminators = {}
        self.mapping = {}

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        X_bin_train = binarize_dataset(X_train, self.bits_per_input)
        self._create_mapping(X_bin_train.shape[1])

        classes = np.unique(y_train)
        for c in classes:
            x = X_bin_train[y_train == c]
            discriminator = DictDiscriminator(self.n_nodes, self.mapping)
            discriminator.train(x)
            self.discriminators[c] = discriminator

    def _create_mapping(self, input_size: int):
        rng = np.random.default_rng()

        for node in range(self.n_nodes):
            self.mapping[node] = rng.integers(0, input_size, size=self.n_tuples)
