from random import sample
from typing import cast

from matplotlib import pyplot as plt
from numpy import ndarray
from torch import Tensor, no_grad
from torch.nn import Module
from torchattacks.attack import Attack

from .base_model import BaseModel
from .helpers import denormalize
from .types import DataLoaderTensor, Normalization


def to_numpy(tensor: Tensor) -> ndarray:
    """Convert a PyTorch tensor to a NumPy array."""

    return tensor.cpu().numpy().transpose(1, 2, 0)


def show_perturbation_example(Model: BaseModel, attack: Attack, rows: int = 3) -> None:
    """Display a perturbation example using `matplotlib`."""

    nn_module: Module = Model.get()

    test_loader: DataLoaderTensor = Model.test_loader
    normalization: Normalization = Model.normalization

    device = next(nn_module.parameters()).device

    images, labels = next(iter(test_loader))
    indices = sample(range(len(images)), k=min(rows, len(images)))

    model_name = Model.__class__.__name__
    attack_name = attack.__class__.__name__
    title = f"Model {model_name} | Attack {attack_name}"

    axes: ndarray
    _, axes = plt.subplots(nrows=rows, ncols=3, num=title, figsize=(9, 3 * rows))  # pyright: ignore[reportUnknownMemberType]

    _ = nn_module.eval()

    for row_index, sample_index in enumerate(indices):
        image = images[[sample_index]].to(device)
        label = labels[[sample_index]].to(device)

        adv_image = cast(Tensor, attack(image, label))

        orig_display = denormalize(image[0].cpu(), normalization)
        adv_display = denormalize(adv_image[0].cpu(), normalization)
        diff_display = (adv_display - orig_display).abs()
        diff_display = diff_display / diff_display.max().clamp(min=1e-8)

        with no_grad():
            prediction_clean = nn_module(image).argmax(dim=1).item()
            prediction_dirty = nn_module(adv_image).argmax(dim=1).item()

        axes[row_index, 0].imshow(to_numpy(orig_display), cmap=Model.CMAP)
        axes[row_index, 1].imshow(to_numpy(adv_display), cmap=Model.CMAP)
        axes[row_index, 2].imshow(to_numpy(diff_display), cmap=Model.CMAP)

        axes[row_index, 0].set_ylabel(f"Sample {sample_index}: {label.item()}")
        axes[row_index, 0].set_title(f"Prediction (clean): {prediction_clean}")
        axes[row_index, 1].set_title(f"Prediction ({attack_name}): {prediction_dirty}")
        axes[row_index, 2].set_title("Perturbation")

    plt.tight_layout()
    plt.show()  # pyright: ignore[reportUnknownMemberType]
