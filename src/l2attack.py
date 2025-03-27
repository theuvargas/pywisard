import numpy as np
from Wisard import Wisard, BloomWisard, DictWisard
from utils import get_mnist
from BloomFilter import bloom_filter_parameters
import time
import matplotlib.pyplot as plt


def main():
    X_train, y_train, X_test, y_test = get_mnist()

    size, hashes = bloom_filter_parameters(6000, 0.05)
    bloom_wisard = BloomWisard(
        n_tuples=48,
        n_nodes=150,
        bits_per_input=6,
        bloom_size=size,
        num_hashes=hashes,
        dtype=bool,
        withBleaching=False,
        n_jobs=1,
    )
    bloom_wisard2 = BloomWisard(
        n_tuples=48,
        n_nodes=50,
        bits_per_input=4,
        bloom_size=size,
        num_hashes=hashes,
        dtype=bool,
        withBleaching=False,
        n_jobs=1,
    )
    dict_wisard = DictWisard(n_tuples=48, n_nodes=50, bits_per_input=4, n_jobs=1)

    print("Training...")
    start = time.time()
    bloom_wisard.train(X_train, y_train)
    print(f"Training time: {time.time() - start}")

    i = 0
    adversary = generate_adversarial_example(
        bloom_wisard,
        X_test[i].reshape(1, -1),
        y_test[i],
        epsilon=2,
        # target=0,  # Optional target for targeted attack
    )

    print(np.linalg.norm(X_test[i] - adversary))

    plot_number(X_test[i], "original")
    plot_number(adversary, "adversary")

    print("Training...")
    start = time.time()
    bloom_wisard2.train(X_train, y_train)
    print(f"Training time: {time.time() - start}")

    print(
        f"Bloom Wisard 2 original prediction: {bloom_wisard2.predict_proba(X_test[i].reshape(1, -1))}"
    )
    print(
        f"Bloom Wisard 2 adversarial prediction: {bloom_wisard2.predict_proba(adversary.reshape(1, -1))}"
    )

    print("Training...")
    start = time.time()
    dict_wisard.train(X_train, y_train)
    print(f"Training time: {time.time() - start}")
    print(
        f"Other Wisard original prediction: {dict_wisard.predict_proba(X_test[i].reshape(1, -1))}"
    )
    print(
        f"Other Wisard adversarial prediction: {dict_wisard.predict_proba(adversary.reshape(1, -1))}"
    )


def plot_number(image, name):
    plt.imshow(image.reshape(28, 28), cmap="gray")
    plt.savefig(f"{name}.png")
    plt.close()


def generate_adversarial_example(
    model,
    image: np.ndarray,
    label,
    epsilon=0.1,
    step_size=1,
    step_decay=0.9999,
    target=None,
):
    original_image = np.copy(image)
    add_noise = create_l2_noise_generator(original_image, epsilon)
    adv_image, _ = simulated_annealing(
        model,
        original_image,
        label,
        add_noise,
        target=target,
        step_size=step_size,
        step_decay=step_decay,
    )

    print("============================================")
    print("Original prediction:", model.predict_proba(original_image))
    print("Adversarial prediction:", model.predict_proba(adv_image))
    return adv_image


def fitness(model: Wisard, image, label, target=None):
    counts = model.predict_proba(image)[0]  # Assume returns dict {class: count}
    total = sum(counts.values())

    if total == 0:
        return 0  # Edge case: no discriminators activated

    # Convert counts to probabilities
    proba = {cls: cnt / total for cls, cnt in counts.items()}
    original_prob = proba.get(label, 0)

    if target is not None:
        # Targeted attack: maximize target vs original
        other_probs = [prob for cls, prob in proba.items() if cls != target]
        max_other_prob = max(other_probs)
        target_prob = proba.get(target, 0)
        return target_prob - max_other_prob
    else:
        # Untargeted attack: maximize best competitor vs original
        other_probs = [prob for cls, prob in proba.items() if cls != label]
        max_other_prob = max(other_probs)
        if not other_probs:
            return 0
        return max_other_prob - original_prob


def simulated_annealing(
    model: Wisard,
    image,
    label,
    add_noise,
    step_size,
    step_decay,
    target=None,
    n_iter=20_000,
    with_early_exit=False,
):
    current_image = np.copy(image)
    current_fitness = fitness(model, current_image, label, target)

    best_image = np.copy(current_image)
    best_fitness = current_fitness

    temperature = 10.0
    cooling_rate = 0.999
    min_temp = 0.001

    for i in range(n_iter):
        if with_early_exit and best_fitness > 0:
            break

        candidate_image = add_noise(current_image, step_size)
        step_size = max(step_size * step_decay, 0.1)

        candidate_fitness = fitness(model, candidate_image, label, target)

        delta = candidate_fitness - current_fitness

        max_exp = 20  # Prevent overflow (e^20 ≈ 4.8e8)
        raw_exp = delta / temperature
        clipped_exp = np.clip(raw_exp, -max_exp, max_exp)
        probability = np.exp(clipped_exp) if temperature > 1e-10 else 0

        if delta >= 0 or np.random.random() < probability:
            current_image = candidate_image
            current_fitness = candidate_fitness

            if current_fitness > best_fitness:
                best_image = np.copy(current_image)
                best_fitness = current_fitness

        temperature = max(temperature * cooling_rate, min_temp)

        if i % 100 == 0:
            print(
                f"[{i}] {best_fitness:.4f} ({temperature:.4f}deg), step_size: {step_size:.4f}, Prob: {probability:.4f}",
            )

    return best_image, best_fitness


def create_l2_noise_generator(original_image, epsilon):
    def add_noise(current_image, step_size):
        delta = current_image - original_image

        perturbation = np.random.randn(*delta.shape)
        perturbation_norm = np.linalg.norm(perturbation)
        if perturbation_norm > 0:
            perturbation = (perturbation / perturbation_norm) * step_size
        else:
            perturbation = np.zeros_like(perturbation)

        new_delta = delta + perturbation
        new_delta_norm = np.linalg.norm(new_delta)
        if new_delta_norm > epsilon:
            new_delta = new_delta * (epsilon / new_delta_norm)

        candidate_image = np.clip(original_image + new_delta, 0, 1)
        return candidate_image

    return add_noise


if __name__ == "__main__":
    main()
