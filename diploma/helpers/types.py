from pathlib import Path
from typing import Protocol

from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torchattacks.attack import Attack

type DataLoaderTensor = DataLoader[tuple[Tensor, Tensor]]
type Normalization = tuple[list[float], list[float]]


class DataLoaderFunc(Protocol):
    def __call__(
        self,
        batch_size: int = -1,
        num_workers: int = -1,
        data_dir: Path | str = "./data",
    ) -> tuple[DataLoaderTensor, DataLoaderTensor]: ...


class TorchModelFunc(Protocol):
    def __call__(
        self,
        epochs: int = -1,
        warmup_epochs: int = -1,
        AT: Attack | None = None,
        GDA: float | None = None,
    ) -> Module: ...
