from pywisard.attacks.utils import get_wisard, eval_wisard
import numpy as np


def l2_noise(image, epsilon):
    if epsilon < 0:
        raise ValueError("Epsilon must be non-negative.")

    image = np.asarray(image)

    eta = np.random.randn(*image.shape)
    norm_eta = np.linalg.norm(eta)

    if norm_eta == 0:
        noise = np.zeros_like(image)
    else:
        noise = epsilon * eta / norm_eta

    noisy_image = image + noise

    noisy_image = np.clip(noisy_image, 0, 1)

    return noisy_image


def linf_noise(image, epsilon):
    if epsilon < 0:
        raise ValueError("Epsilon must be non-negative.")
    image = np.asarray(image)

    noise = np.random.choice([-epsilon, epsilon], size=image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image


def main():
    print("Bloom Wisard")
    bloom_wisard, X_test, y_test = get_wisard("bloom", with_eval=True)
    print("\nDict Wisard")
    dict_wisard, _, _ = get_wisard("dict", with_eval=True)

    X_noisy_linf = np.array([linf_noise(image, 0.1) for image in X_test])
    X_noisy_l2 = np.array([l2_noise(image, 1.58) for image in X_test])

    print("\n===== LINF =====\n")
    print("Bloom Wisard")
    eval_wisard(bloom_wisard, X_noisy_linf, y_test)
    print("\nDict Wisard")
    eval_wisard(dict_wisard, X_noisy_linf, y_test)

    print("\n===== L2 =====\n")
    print("Bloom Wisard")
    eval_wisard(bloom_wisard, X_noisy_l2, y_test)
    print("\nDict Wisard")
    eval_wisard(dict_wisard, X_noisy_l2, y_test)


if __name__ == "__main__":
    main()
