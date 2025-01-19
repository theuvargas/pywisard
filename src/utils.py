import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np


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


def binarize_input(input: float, output_size: int, min_val: float, max_val: float):
    binarized = np.ones(output_size)
    thermometer = np.linspace(min_val, max_val, output_size + 1, endpoint=False)[1:]

    print(thermometer)
    for i in range(output_size - 1, 1, -1):
        if input > thermometer[i]:
            break
        binarized[i] = 0

    return binarized


def binarize_dataset(dataset: np.ndarray, bits_per_input: int):
    min_val = dataset.min()
    max_val = dataset.max()

    thermometer = np.linspace(min_val, max_val, bits_per_input)

    reshaped_data = dataset.reshape(dataset.shape[0], dataset.shape[1], 1)
    reshaped_thermometer = thermometer.reshape(1, 1, -1)

    binarized = (reshaped_data >= reshaped_thermometer).astype(np.float32)

    return binarized.reshape(dataset.shape[0], -1)
