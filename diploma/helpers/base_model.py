"""Abstract base model class."""

from abc import ABC, abstractmethod
from functools import cache, partial
from typing import TYPE_CHECKING, ClassVar

from torchattacks import CW, FGSM, JSMA
from torchattacks import DeepFool as DF

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.nn import Module
    from torchattacks.attack import Attack

    from .types import DataLoaderTensor, Normalization


class BaseModel(ABC):
    """Abstract base class for MNIST and CIFAR10 models."""

    FGSM_EPS: float = 0.25
    JSMA_THETA: float = 1.0
    JSMA_GAMMA: float = 0.1
    DF_STEPS: int = 50
    DF_OVERSHOOT: float = 0.02
    CW_C: float = 5
    CW_STEPS: int = 50
    CW_LR: float = 0.1

    ATTACKS: ClassVar[dict[str, tuple[bool, Callable[..., Attack]]]] = {
        "FGSM": (False, partial(FGSM, eps=FGSM_EPS)),
        "JSMA": (True,  partial(JSMA, theta=JSMA_THETA, gamma=JSMA_GAMMA)),
          "DF": (True,  partial(DF,   steps=DF_STEPS,   overshoot=DF_OVERSHOOT)),
          "CW": (False, partial(CW,   c=CW_C, steps=CW_STEPS, lr=CW_LR)),
    }  # fmt: off
    """
    # Dictionary containing:
    * `bool`: whether the attack is slow or not.
    * `Callable[..., Attack]`: `Callable` that returns `Attack`.
    """

    CMAP: str | None = None
    """
    Color map to use for visualizations. Relevant values are:
    * `None` - for RGB images
    * `"gray"` - for grayscale images
    """

    def __init__(self, name: str, normalization: Normalization) -> None:
        """Initialize the base model with the given name and normalization."""
        self.name: str = name
        self.normalization: Normalization = normalization

    @abstractmethod
    @cache
    def get(
        self,
        epochs: int,
        warmup_epochs: int,
        AT: Attack | None = None,
        GDA: float | None = None,
    ) -> Module:
        """Return the model as a `torch.nn.Module`."""
        ...

    def get_attack(self, attack_name: str) -> tuple[Attack, bool]:
        """Return the attack and whether it is slow."""
        is_slow, attack_callable = self.ATTACKS[attack_name]

        attack = attack_callable(model=self.get())
        attack.set_normalization_used(*self.normalization)

        return attack, is_slow

    @property
    @abstractmethod
    def loaders(self) -> tuple[DataLoaderTensor, DataLoaderTensor]:
        """Return the train and test loaders."""

    @property
    def train_loader(self) -> DataLoaderTensor:
        """Return the train loader."""
        return self.loaders[0]

    @property
    def test_loader(self) -> DataLoaderTensor:
        """Return the test loader."""
        return self.loaders[1]
