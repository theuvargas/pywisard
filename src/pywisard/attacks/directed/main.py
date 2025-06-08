import random
from pywisard.attacks.directed.utils import (
    pixel_importance,
    flip_distances,
)
from pywisard.attacks.utils import get_wisard
import numpy as np
from pywisard.utils import binarize_dataset, save_number
from pywisard.attacks import config
from math import inf
import heapq


def whitebox_attack():
    wisard, X_test, y_test = get_wisard("dict", with_eval=False)

    i = 0
    image = X_test[i]

    distances = flip_distances(image, np.array([0.5]))
    abs_distances = np.abs(distances)

    y_true = y_test[i]
    mappings = wisard.mapping

    true_discriminator = wisard.discriminators[y_true].nodes  # [0].memory

    print(wisard.predict_proba(np.array([image])))

    adversary = image.copy()

    for mapping, node in zip(mappings.values(), true_discriminator):
        for tup in node.memory.keys():
            bin_adv = binarize_dataset(np.array([adversary]), config.BITS_PER_INPUT)[0]
            true_bin_values = tuple(bin_adv[mapping])
            if tup == true_bin_values:  # match
                # find the pixel index with minimum absolute distance in current mapping
                min_dist_idx = mapping[np.argmin(abs_distances[mapping])]

                if np.linalg.norm(adversary - image) > config.EPSILON:
                    continue
                abs_distances[min_dist_idx] = inf

                # flip the pixel value
                adversary[min_dist_idx] += distances[min_dist_idx]

    print(np.linalg.norm(adversary - image))
    print(wisard.predict_proba(np.array([adversary])))


def whitebox_attack2():
    wisard, X_test, y_test = get_wisard("dict", with_eval=False)

    i = 0
    image = X_test[i]
    y_true = y_test[i]

    print("Original classification:")
    print(wisard.predict_proba(np.array([image]))[0])

    true_discriminator = wisard.discriminators[y_true].nodes
    seen_patterns = {}
    for map, node in zip(wisard.mapping.values(), true_discriminator):
        map = tuple(map)
        seen_patterns[map] = tuple(node.memory.keys())

    adversary = image.copy()

    seen_patterns = {
        map_key: np.array(patterns) for map_key, patterns in seen_patterns.items()
    }

    should_break = False
    i = 0
    distances = flip_distances(adversary, np.array([0.5]))
    while not should_break:
        if i == 10:
            break
        i += 1
        for mapping in wisard.mapping.values():
            map = tuple(mapping)

            bin_adv = binarize_dataset(np.array([adversary]), config.BITS_PER_INPUT)[0]
            true_bin_values = bin_adv[[map]][0]

            pattern_match = (seen_patterns[map] == true_bin_values).all(axis=1).any()

            if pattern_match:
                index = random.choice(map)

                # flip the pixel value
                adversary[index] += distances[index]

                if np.linalg.norm(adversary - image) > config.EPSILON:
                    should_break = True
                    adversary[index] -= distances[index]
                    break

    print("Noise budget used: ", np.linalg.norm(adversary - image))
    print("Adversarial classification:")
    print(wisard.predict_proba(np.array([adversary]))[0])
    save_number(adversary)


def whitebox_attack2_all():
    wisard, X_test, y_test = get_wisard("dict", with_eval=False)

    correct = 0
    for k in range(len(X_test)):
        if k % 1000 == 0:
            print(k)
        image = X_test[k]
        y_true = y_test[k]

        if wisard.predict(np.array([image]))[0] != y_true:
            continue

        true_discriminator = wisard.discriminators[y_true].nodes
        seen_patterns = {}
        for map, node in zip(wisard.mapping.values(), true_discriminator):
            map = tuple(map)
            seen_patterns[map] = tuple(node.memory.keys())

        adversary = image.copy()

        seen_patterns = {
            map_key: np.array(patterns) for map_key, patterns in seen_patterns.items()
        }

        should_break = False
        i = 0
        distances = flip_distances(adversary, np.array([0.5]))
        while not should_break:
            if i == 10:
                break
            i += 1
            for mapping in wisard.mapping.values():
                map = tuple(mapping)

                bin_adv = binarize_dataset(
                    np.array([adversary]), config.BITS_PER_INPUT
                )[0]
                true_bin_values = bin_adv[[map]][0]

                pattern_match = (
                    (seen_patterns[map] == true_bin_values).all(axis=1).any()
                )

                if pattern_match:
                    index = random.choice(map)

                    # flip the pixel value
                    adversary[index] += distances[index]

                    if np.linalg.norm(adversary - image) > config.EPSILON:
                        should_break = True
                        adversary[index] -= distances[index]
                        break

        if wisard.predict(np.array([adversary]))[0] == y_true:
            correct += 1

    print("Accuracy: ", correct / len(X_test))


def whitebox_targeted():
    wisard, X_test, y_test = get_wisard("dict", with_eval=False)

    i = 0
    image = X_test[i]
    y_true = y_test[i]
    y_target = 9

    print("Original classification:")
    print(wisard.predict_proba(np.array([image]))[0])

    target_discriminator = wisard.discriminators[y_target].nodes
    seen_patterns = {}
    for map, node in zip(wisard.mapping.values(), target_discriminator):
        map = tuple(map)
        seen_patterns[map] = tuple(node.memory.keys())

    adversary = image.copy()

    seen_patterns = {
        map_key: np.array(patterns) for map_key, patterns in seen_patterns.items()
    }


def main():
    whitebox_attack2_all()

    # whitebox_attack()
    # wisard, X_test, y_test = get_bloom_wisard()
    #
    # image = X_test[0]
    #
    # print(wisard.predict_proba(np.array([image])))
    #
    # importance = pixel_importance(wisard, image)
    # distances = flip_distances(image, np.array([0.5]))
    # positive_distances = np.abs(distances)
    # print(positive_distances)


if __name__ == "__main__":
    main()
