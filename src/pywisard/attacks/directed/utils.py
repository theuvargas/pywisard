from pywisard.Wisard import Wisard
import numpy as np
from numpy.random import default_rng


def pixel_importance(model: Wisard, image: np.ndarray) -> np.ndarray:
    classification = np.array(list(model.predict_proba(np.array([image]))[0].values()))
    test_images = []

    for pixel in range(image.shape[0]):
        new_image = image.copy()

        if image[pixel] < 0.5:
            new_image[pixel] += 0.5
        else:
            new_image[pixel] -= 0.5

        test_images.append(new_image)

    y_test = model.predict_proba(np.array(test_images))

    diffs = []
    for score in y_test:
        score = np.array(list(score.values()))

        diff = np.abs(classification - score)
        diffs.append(diff.sum())

    return np.array(diffs)


# def tuple_importance(
#     model: Wisard, image: np.ndarray, tuple_len: int, num_tuples: int
# ) -> np.ndarray:
#     classification = np.array(list(model.predict_proba(np.array([image]))[0].values()))
#     test_images = {}
#
#     rng = default_rng()
#
#     for _ in range(num_tuples):
#         indices = rng.choice(np.arange(image.shape[0]), size=tuple_len, replace=False)
#         for i in range(tuple_len):
#             for j in range(tuple_len):
#                 new_image = image.copy()
#                 if pixel < 0.5:
#                     new_image[pixel] += 0.5
#                 else:
#                     new_image[pixel] -= 0.5
#
#                 test_images[tuple(indices)] = new_image


def flip_distances(image: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    distances = np.zeros_like(image)
    for i, pixel in enumerate(image):
        closest = np.argmin(np.abs(thresholds - pixel))
        distances[i] = thresholds[closest] - pixel

        # if a pixel is above or at a threshold, it needs to go below
        # it to change its binarization
        if distances[i] <= 0:
            distances[i] -= 0.001
    return distances
