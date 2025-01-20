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
        print("finished binarizing")
        self._create_mapping(X_bin_train.shape[1])

        classes = np.unique(y_train)
        for c in classes:
            x = X_bin_train[y_train == c]
            discriminator = DictDiscriminator(self.n_nodes, self.mapping)
            discriminator.train(x)
            self.discriminators[c] = discriminator

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_bin_test = binarize_dataset(X_test, self.bits_per_input)
        y_pred = np.zeros(X_bin_test.shape[0])

        for i, x in enumerate(X_bin_test):
            scores = {}
            bleaching = 1
            while True:
                for c, discriminator in self.discriminators.items():
                    scores[c] = discriminator.get_response(x, bleaching)
                max_val = max(scores.values())
                if max_val == 0:  # prevents infinite loop
                    bleaching -= 10
                    for c, discriminator in self.discriminators.items():
                        scores[c] = discriminator.get_response(x, bleaching)
                    break
                num_max = 0
                for score in scores.values():
                    if score == max_val:
                        num_max += 1
                        if num_max > 1:
                            break
                if num_max == 1:
                    break
                bleaching += 10

            y_pred[i] = max(scores, key=scores.get)

        return y_pred

    def _create_mapping(self, input_size: int):
        rng = np.random.default_rng()

        for node in range(self.n_nodes):
            self.mapping[node] = rng.integers(0, input_size, size=self.n_tuples)
