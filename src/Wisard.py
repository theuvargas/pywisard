from abc import ABC, abstractmethod
import numpy as np
from Discriminator import BloomDiscriminator, DictDiscriminator
from utils import binarize_dataset


class Wisard(ABC):
    def __init__(self, n_tuples: int, n_nodes: int, bits_per_input: int):
        self.n_tuples = n_tuples
        self.n_nodes = n_nodes
        self.bits_per_input = bits_per_input
        self.discriminators = {}
        self.mapping = {}

    @abstractmethod
    def _create_discriminator(
        self, mapping: dict[int, np.ndarray]
    ) -> BloomDiscriminator | DictDiscriminator:
        pass

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_bin_train = binarize_dataset(X_train, self.bits_per_input)
        self._create_mapping(X_bin_train.shape[1])

        classes = np.unique(y_train)
        for c in classes:
            x = X_bin_train[y_train == c]
            discriminator = self._create_discriminator(self.mapping)
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

                if self._should_break(scores):
                    break

                bleaching = self._adjust_bleaching(scores, bleaching)

            y_pred[i] = max(scores, key=scores.get)

        return y_pred

    def _create_mapping(self, input_size: int) -> None:
        rng = np.random.default_rng()
        for node in range(self.n_nodes):
            self.mapping[node] = rng.integers(0, input_size, size=self.n_tuples)

    def _should_break(self, scores: dict) -> bool:
        max_val = max(scores.values())
        if max_val == 0:
            return True

        num_max = sum(1 for score in scores.values() if score == max_val)
        return num_max == 1

    def _adjust_bleaching(self, scores: dict, current_bleaching: int) -> int:
        max_val = max(scores.values())
        if max_val == 0:
            return current_bleaching - 10
        return current_bleaching + 10


class DictWisard(Wisard):
    def _create_discriminator(
        self, mapping: dict[int, np.ndarray]
    ) -> DictDiscriminator:
        return DictDiscriminator(self.n_nodes, mapping)


class BloomWisard(Wisard):
    def __init__(
        self,
        n_tuples: int,
        n_nodes: int,
        bits_per_input: int,
        bloom_size: int,
        num_hashes: int,
    ):
        super().__init__(n_tuples, n_nodes, bits_per_input)
        self.bloom_size = bloom_size
        self.num_hashes = num_hashes

    def _create_discriminator(
        self, mapping: dict[int, np.ndarray]
    ) -> BloomDiscriminator:
        return BloomDiscriminator(
            self.n_nodes,
            mapping,
            self.bloom_size,
            self.num_hashes,
        )
