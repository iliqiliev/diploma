"""CIFAR-10 dataset utilities."""

from typing import TYPE_CHECKING, cast

from torch import float32
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

if TYPE_CHECKING:
    from pathlib import Path

    from diploma.helpers import DataLoaderTensor, Normalization

CIFAR10_CLASSES: int = 10
CIFAR10_RESOLUTION: int = 32

CIFAR10_MEAN: list[float] = [0.49139968, 0.48215827, 0.44653124]
CIFAR10_STD: list[float] = [0.24703233, 0.24348505, 0.26158768]
CIFAR10_NORMALIZATION: Normalization = (CIFAR10_MEAN, CIFAR10_STD)


def get_cifar10_loaders(
    batch_size: int = 4096,
    num_workers: int = 8,
    data_dir: Path | str = "./data/CIFAR10",
) -> tuple[DataLoaderTensor, DataLoaderTensor]:
    """Return the CIFAR-10 train and test data loaders."""
    train_transform = v2.Compose(
        [
            v2.RandomCrop(CIFAR10_RESOLUTION, padding=4),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(float32, scale=True),
            v2.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ],
    )
    train_set = CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    train_loader = cast(
        "DataLoaderTensor",
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
            v2.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ],
    )
    test_set = CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )
    test_loader = cast(
        "DataLoaderTensor",
        DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    )

    return train_loader, test_loader
