from pywisard.Wisard import BloomWisard, DictWisard
from pywisard.BloomFilter import bloom_filter_parameters
from pywisard.utils import get_mnist
from pywisard.attacks.config import (
    N_TUPLES,
    N_NODES,
    BITS_PER_INPUT,
    WITH_BLEACHING,
    N_JOBS,
    DTYPE,
)
import time


def get_wisard(model: str, with_eval=False):
    wisard = None

    if model == "bloom":
        size, hashes = bloom_filter_parameters(6000, 0.05)
        wisard = BloomWisard(
            n_tuples=N_TUPLES,
            n_nodes=N_NODES,
            bits_per_input=BITS_PER_INPUT,
            bloom_size=size,
            num_hashes=hashes,
            dtype=DTYPE,
            withBleaching=WITH_BLEACHING,
            n_jobs=N_JOBS,
        )
    elif model == "dict":
        wisard = DictWisard(
            n_tuples=N_TUPLES,
            n_nodes=N_NODES,
            bits_per_input=BITS_PER_INPUT,
            dtype=DTYPE,
            withBleaching=WITH_BLEACHING,
            n_jobs=N_JOBS,
        )
    else:
        raise ValueError("Model must be 'bloom' or 'dict'")

    X_train, y_train, X_test, y_test = get_mnist()

    print("Training Wisard...")
    start = time.time()
    wisard.train(X_train, y_train)
    print(f"Training time: {time.time() - start}")

    if with_eval:
        eval_wisard(wisard, X_test, y_test)

    return wisard, X_test, y_test


def eval_wisard(wisard, X_test, y_test):
    print("Evaluating Wisard...")
    start = time.time()
    y_pred = wisard.predict(X_test)
    print(f"Predict time: {time.time() - start}")

    accuracy = (y_pred == y_test).sum() / len(y_test)
    print(f"Accuracy: {accuracy}")
