from pywisard.attacks.utils import eval_wisard
from pywisard.Wisard import BloomWisard, DictWisard
from pywisard.BloomFilter import bloom_filter_parameters
from pywisard.utils import get_mnist
import time


def main():
    model = "dict"
    n_tuples = 48
    n_nodes = 50
    bits_per_input = 6
    dtype = bool
    with_bleaching = False

    if model == "bloom":
        size, hashes = bloom_filter_parameters(6000, 0.05)
        wisard = BloomWisard(
            n_tuples=n_tuples,
            n_nodes=n_nodes,
            bits_per_input=bits_per_input,
            bloom_size=size,
            num_hashes=hashes,
            dtype=dtype,
            withBleaching=with_bleaching,
            n_jobs=-1,
        )
    elif model == "dict":
        wisard = DictWisard(
            n_tuples=n_tuples,
            n_nodes=n_nodes,
            bits_per_input=bits_per_input,
            dtype=dtype,
            withBleaching=with_bleaching,
            n_jobs=1,
        )
    else:
        raise ValueError("Model must be 'bloom' or 'dict'")

    print("bits_per_input:", bits_per_input)
    X_train, y_train, X_test, y_test = get_mnist()

    print("Training Wisard...")
    start = time.time()
    wisard.train(X_train, y_train)
    print(f"Training time: {time.time() - start}")

    eval_wisard(wisard, X_test, y_test)


if __name__ == "__main__":
    main()
