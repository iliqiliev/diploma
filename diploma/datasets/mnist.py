from pathlib import Path
from typing import cast

from torch import float32
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import v2

from ..helpers import DataLoaderTensor, Normalization

MNIST_CLASSES: int = 10
MNIST_RESOLUTION: int = 28

MNIST_MEAN: list[float] = [0.1307]
MNIST_STD: list[float] = [0.3081]
MNIST_NORMALIZATION: Normalization = (MNIST_MEAN, MNIST_STD)


def get_mnist_loaders(
    batch_size: int = 512,
    num_workers: int = 8,
    data_dir: Path | str = "./data",
) -> tuple[DataLoaderTensor, DataLoaderTensor]:
    """Returns the MNIST train and test data loaders."""

    train_transform = v2.Compose(
        [
            v2.RandomCrop(MNIST_RESOLUTION, padding=3),
            v2.ToImage(),
            v2.ToDtype(float32, scale=True),
            v2.Normalize(mean=MNIST_MEAN, std=MNIST_STD),
        ]
    )
    train_set = MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    train_loader = cast(
        DataLoaderTensor,
        DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
    )

    test_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(float32, scale=True),
            v2.Normalize(mean=MNIST_MEAN, std=MNIST_STD),
        ]
    )
    test_set = MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )
    test_loader = cast(
        DataLoaderTensor,
        DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    )

    return train_loader, test_loader
