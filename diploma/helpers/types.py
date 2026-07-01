"""Typing helpers."""

from typing import TYPE_CHECKING, Protocol

from torch import Tensor
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from pathlib import Path

    from torch.nn import Module
    from torchattacks.attack import Attack

type DataLoaderTensor = DataLoader[tuple[Tensor, Tensor]]
type Normalization = tuple[list[float], list[float]]


class DataLoaderFunc(Protocol):
    """Protocol for data loader functions."""

    def __call__(
        self,
        batch_size: int = -1,
        num_workers: int = -1,
        data_dir: Path | str = "./data",
    ) -> tuple[DataLoaderTensor, DataLoaderTensor]:
        """Call signature."""
        ...


class TorchModelFunc(Protocol):
    """Protocol for torch model functions."""

    def __call__(
        self,
        epochs: int = -1,
        warmup_epochs: int = -1,
        AT: Attack | None = None,
        GDA: float | None = None,
    ) -> Module:
        """Call signature."""
        ...
