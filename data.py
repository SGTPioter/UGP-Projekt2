from dataclasses import dataclass

import numpy as np
from tensorflow import keras


IMAGE_SHAPE = (32, 32, 3)
CLASS_NAMES = (
    "samolot", "samochód", "ptak", "kot", "jeleń",
    "pies", "żaba", "koń", "statek", "ciężarówka",
)


@dataclass(frozen=True)
class ImageSplit:
    images: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class Cifar10Data:
    train: ImageSplit
    validation: ImageSplit
    test: ImageSplit

    @classmethod
    def load(
        cls, train_per_class: int, val_per_class: int, test_per_class: int, seed: int
    ) -> "Cifar10Data":
        (train_images, train_labels), (test_images, test_labels) = (
            keras.datasets.cifar10.load_data()
        )
        train_labels = train_labels.ravel()
        test_labels = test_labels.ravel()
        rng = np.random.default_rng(seed)
        train_indices, val_indices, test_indices = [], [], []
        for label in range(len(CLASS_NAMES)):
            indices = rng.permutation(np.flatnonzero(train_labels == label))
            train_indices.extend(indices[:train_per_class])
            val_indices.extend(indices[train_per_class:train_per_class + val_per_class])
            indices = rng.permutation(np.flatnonzero(test_labels == label))
            test_indices.extend(indices[:test_per_class])

        def select(images, labels, indices):
            shuffled = rng.permutation(indices)
            return ImageSplit(images[shuffled], labels[shuffled])

        return cls(
            train=select(train_images, train_labels, train_indices),
            validation=select(train_images, train_labels, val_indices),
            test=select(test_images, test_labels, test_indices),
        )
