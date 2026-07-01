"""Datasets module."""

from .cifar10 import CIFAR10_CLASSES, CIFAR10_NORMALIZATION, get_cifar10_loaders
from .mnist import MNIST_CLASSES, MNIST_NORMALIZATION, get_mnist_loaders

__all__ = [
    "CIFAR10_CLASSES",
    "CIFAR10_NORMALIZATION",
    "MNIST_CLASSES",
    "MNIST_NORMALIZATION",
    "get_cifar10_loaders",
    "get_mnist_loaders",
]
