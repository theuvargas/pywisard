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


# def binarize_dataset(dataset: np.ndarray, output_size: int):
#     min_val = dataset.min()
#     max_val = dataset.max()
#
#     binarized_dataset = np.zeros((dataset.shape[0], dataset.shape[1] * output_size))
#
#     for i in range(dataset.shape[0]):
#         for j in range(dataset.shape[1]):
#             binarized_dataset[i, j * output_size : (j + 1) * output_size] = (
#                 binarize_input(dataset[i, j], output_size, min_val, max_val)
#             )
#
#     return binarized_dataset


def binarize_dataset(dataset: np.ndarray, output_size: int):
    min_val = dataset.min()
    max_val = dataset.max()

    # Create thermometer levels for comparison
    thermometer = np.linspace(min_val, max_val, output_size)

    # Reshape dataset and thermometer for broadcasting
    # dataset shape: (n_samples, n_features, 1)
    # thermometer shape: (1, 1, output_size)
    reshaped_data = dataset.reshape(dataset.shape[0], dataset.shape[1], 1)
    reshaped_thermometer = thermometer.reshape(1, 1, -1)

    # Compare in one operation
    # This creates a (n_samples, n_features, output_size) boolean array
    binarized = (reshaped_data >= reshaped_thermometer).astype(np.float32)

    # Reshape to match original output format
    return binarized.reshape(dataset.shape[0], -1)
