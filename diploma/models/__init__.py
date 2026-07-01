"""Models package."""

from .cnn_mnist import Mnist
from .resnet18_cifar10 import Cifar10

__all__ = [
    "Cifar10",
    "Mnist",
]
