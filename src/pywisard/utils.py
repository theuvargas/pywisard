import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import random


def get_mnist():
    train_dataset = dsets.MNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=True
    )
    test_dataset = dsets.MNIST(
        root="./data", train=False, transform=transforms.ToTensor()
    )

    train_images = []
    train_labels = []
    for image, label in train_dataset:
        train_images.append(image.numpy().flatten())
        train_labels.append(label)

    test_images = []
    test_labels = []
    for image, label in test_dataset:
        test_images.append(image.numpy().flatten())
        test_labels.append(label)

    return (
        np.array(train_images),
        np.array(train_labels),
        np.array(test_images),
        np.array(test_labels),
    )


def binarize_dataset(dataset: np.ndarray, bits_per_input: int, min_val=0, max_val=1):
    thermometer = np.linspace(min_val, max_val, bits_per_input + 1, endpoint=False)[1:]

    dataset_expanded = dataset.reshape(dataset.shape[0], dataset.shape[1], 1)
    thermometer_expanded = thermometer.reshape(1, 1, -1)

    comparisons = dataset_expanded >= thermometer_expanded

    binarized = np.ones(
        (dataset.shape[0], dataset.shape[1] * bits_per_input), dtype=np.int8
    )
    binarized.reshape(dataset.shape[0], dataset.shape[1], bits_per_input)[
        :
    ] = comparisons

    return binarized


def save_adversaries(y_pred_wisard_adv, original_labels, successful_advs_np):
    successful_transfer_indices = np.where(y_pred_wisard_adv != original_labels)[0]

    if len(successful_transfer_indices) >= 3:
        # Randomly select 3 indices
        selected_indices = random.sample(list(successful_transfer_indices), 3)

        # Save the selected adversarial images
        for i, idx in enumerate(selected_indices):
            plt.figure()
            plt.imshow(successful_advs_np[idx].reshape(28, 28), cmap="gray")
            # plt.axis("off")
            plt.savefig(f"adv{i+1}.png")
            plt.close()
        print(
            "\nSaved 3 random successful adversarial examples as adv1.png, adv2.png, and adv3.png"
        )
    else:
        print("\nNot enough successful transfers to save 3 images")


def save_number(image):
    plt.figure()
    plt.imshow(image.reshape(28, 28), cmap="gray")
    plt.savefig("adversario")
