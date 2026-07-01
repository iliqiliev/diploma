"""Helper functions."""

from logging import StreamHandler, getLogger
from typing import TYPE_CHECKING, cast

from torch import Tensor, cuda, device, enable_grad, no_grad, tensor
from torch import max as torch_max
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from torch.nn import Module
    from torchattacks.attack import Attack

    from .types import DataLoaderTensor, Normalization


def denormalize(normalized_tensor: Tensor, normalization_used: Normalization) -> Tensor:
    """Convert normalized tensor back to [0, 1] pixel space."""
    mean, std = normalization_used

    mean_tensor = tensor(mean, device=normalized_tensor.device).view(-1, 1, 1)
    std_tensor = tensor(std, device=normalized_tensor.device).view(-1, 1, 1)

    return (normalized_tensor * std_tensor + mean_tensor).clamp(0, 1)


@no_grad()
def get_accuracy(
    model: Module,
    dataloader: DataLoaderTensor,
    attack: Attack | None = None,
    max_samples: int | None = None,
    max_batch_size: int | None = None,
) -> float:
    """Return the accuracy of the model."""
    device = get_torch_device()

    _ = model.eval()
    total = 0
    correct = 0

    if max_batch_size is not None:
        dataloader = DataLoader(
            dataset=dataloader.dataset,
            batch_size=max_batch_size,
            shuffle=False,
            num_workers=dataloader.num_workers,
        )

    inputs: Tensor
    labels: Tensor
    outputs: Tensor

    for _inputs_, _labels_ in dataloader:
        if max_samples is not None and total >= max_samples:
            break

        total += _labels_.size(0)

        inputs, labels = _inputs_.to(device), _labels_.to(device)

        if attack is not None:
            with enable_grad():
                inputs = cast("Tensor", attack(inputs, labels))

        outputs = model(inputs)

        _, predicted = torch_max(outputs.data, 1)
        correct += (predicted == labels).sum().item()

    return correct / total


def get_torch_device(force: str | None = None) -> device:
    """Return the torch device to use. Falls back to CPU if no GPU is available."""
    if force is not None:
        return device(force)

    if cuda.is_available():
        return device("cuda")

    log.warning("get_torch_device(): Falling CPU as fallback")
    return device("cpu")


log = getLogger("diploma")
log.addHandler(StreamHandler())
log.setLevel("INFO")
