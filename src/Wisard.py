from abc import ABC, abstractmethod
import numpy as np
from Discriminator import BloomDiscriminator, DictDiscriminator
from utils import binarize_dataset
from concurrent.futures import ProcessPoolExecutor
import os


class Wisard(ABC):
    def __init__(
        self,
        n_tuples: int,
        n_nodes: int,
        bits_per_input: int,
        dtype: type = int,
        withBleaching: bool = True,
        n_jobs: int = 1,
    ):
        self.n_tuples = n_tuples
        self.n_nodes = n_nodes
        self.bits_per_input = bits_per_input
        self.discriminators = {}
        self.mapping = {}
        self.dtype = dtype
        self.withBleaching = withBleaching
        self.n_jobs = n_jobs

    @abstractmethod
    def _create_discriminator(
        self, mapping: dict[int, np.ndarray]
    ) -> BloomDiscriminator | DictDiscriminator:
        pass

    def _train_single(self, args):
        X_bin_train, y_train, c = args
        x = X_bin_train[y_train == c]
        discriminator = self._create_discriminator(self.mapping)
        discriminator.train(x)
        return c, discriminator

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_bin_train = binarize_dataset(X_train, self.bits_per_input)
        self._create_mapping(X_bin_train.shape[1])
        self.classes = np.unique(y_train)

        args = [(X_bin_train, y_train, c) for c in self.classes]

        if self.n_jobs == 1:
            results = map(self._train_single, args)
        else:
            n_processes = os.cpu_count() or 1 if self.n_jobs == -1 else self.n_jobs
            with ProcessPoolExecutor(max_workers=n_processes) as executor:
                results = executor.map(self._train_single, args)

        self.discriminators.update(dict(results))

    def _process_batch(self, batch_data, method):
        if method == "predict":
            return self._predict_batch(batch_data)
        return self._predict_proba_batch(batch_data)

    def _process_batch_with_method(self, args):
        batch_data, method = args
        return self._process_batch(batch_data, method)

    def _parallel_process(self, X_bin_test: np.ndarray, method: str):
        n_samples = len(X_bin_test)
        n_processes = os.cpu_count() or 1 if self.n_jobs == -1 else self.n_jobs
        batch_size = max(n_samples // n_processes, 1)
        batches = [
            X_bin_test[i : i + batch_size] for i in range(0, n_samples, batch_size)
        ]
        args = [(batch, method) for batch in batches]

        if self.n_jobs == 1:
            results = map(self._process_batch_with_method, args)
        else:
            with ProcessPoolExecutor(max_workers=n_processes) as executor:
                results = executor.map(self._process_batch_with_method, args)

        if method == "predict":
            return np.concatenate(list(results))
        return [score for batch in results for score in batch]

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_bin_test = binarize_dataset(X_test, self.bits_per_input)
        return self._parallel_process(X_bin_test, "predict")

    def predict_proba(self, X_test: np.ndarray) -> list[dict]:
        X_bin_test = binarize_dataset(X_test, self.bits_per_input)
        return self._parallel_process(X_bin_test, "predict_proba")

    def _process_sample(self, x):
        scores = {}
        bleaching = 1
        while True:
            for c, discriminator in self.discriminators.items():
                scores[c] = discriminator.get_response(x, bleaching)
            if self._should_break(scores):
                break
            bleaching = self._adjust_bleaching(scores, bleaching)
        return scores

    def _predict_batch(self, batch_data):
        predictions = np.zeros(len(batch_data))
        for i, x in enumerate(batch_data):
            scores = self._process_sample(x)
            predictions[i] = max(scores, key=scores.get)
        return predictions

    def _predict_proba_batch(self, batch_data):
        predictions = []
        for x in batch_data:
            scores = self._process_sample(x)
            predictions.append(scores)
        return predictions

    def _create_mapping(self, input_size: int) -> None:
        rng = np.random.default_rng()
        for node in range(self.n_nodes):
            self.mapping[node] = rng.integers(0, input_size, size=self.n_tuples)

    def _should_break(self, scores: dict) -> bool:
        if not self.withBleaching:
            return True

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
        return DictDiscriminator(self.n_nodes, mapping, self.dtype, self.withBleaching)


class BloomWisard(Wisard):
    def __init__(
        self,
        n_tuples: int,
        n_nodes: int,
        bits_per_input: int,
        bloom_size: int,
        num_hashes: int,
        dtype: type = int,
        withBleaching: bool = True,
        n_jobs: int = -1,
    ):
        super().__init__(
            n_tuples, n_nodes, bits_per_input, dtype, withBleaching, n_jobs
        )
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
            self.dtype,
            self.withBleaching,
        )
