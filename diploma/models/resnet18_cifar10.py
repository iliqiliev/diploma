"""ResNet18 model for CIFAR10 classification."""

from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, cast

from rich.progress import track
from torch import Tensor, cat, load, randn_like, save
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision.models import ResNet, resnet18

from diploma.datasets import CIFAR10_CLASSES, CIFAR10_NORMALIZATION, get_cifar10_loaders
from diploma.helpers import BaseModel, DataLoaderTensor, get_torch_device, log

if TYPE_CHECKING:
    from torchattacks.attack import Attack


class Cifar10(BaseModel):
    """CIFAR-10 model class."""

    def __init__(self) -> None:
        """Initialize the CIFAR-10 model."""
        super().__init__(
            name="resnet18_cifar10",
            normalization=CIFAR10_NORMALIZATION,
        )

    @cache
    def get(
        self,
        epochs: int = 200,
        warmup_epochs: int = 5,
        AT: Attack | None = None,
        GDA: float | None = None,
    ) -> ResNet:
        """Return the ResNet18 model for CIFAR10 classification.

        Using the default values for `epochs` and `warmup_epochs`
        gives around 83% accuracy.

        The `AT` (Adversarial Training) parameter can be used to
        specify an adversarial training attack for the model to train against.

        The `GDA` (Gaussian Data Augmentation) parameter can be used
        to specify the strength. The value is a `float` type in the range [0, 1].
        """
        torch_device = get_torch_device()

        if AT is not None:
            attack_name = AT.__class__.__name__
            log.info(f"Using AT defence against {attack_name} ...")

            model_path = Path(f"./data/weights/{self.name}_{attack_name}.pth")
            checkpoint_path = Path(f"./data/checkpoints/{self.name}_{attack_name}.pth")

        elif GDA is not None:
            if not 0 <= GDA <= 1:
                error = ValueError("GDA strength must be in the range [0, 1].")
                raise error

            log.info(f"Using GDA defence with strength σ={GDA} ...")

            model_path = Path(f"./data/weights/{self.name}_GDA.pth")
            checkpoint_path = Path(f"./data/checkpoints/{self.name}_GDA.pth")

        else:
            model_path = Path(f"./data/weights/{self.name}.pth")
            checkpoint_path = Path(f"./data/checkpoints/{self.name}.pth")

        model_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        model = resnet18(weights=None, num_classes=CIFAR10_CLASSES)
        model = model.to(torch_device)

        if model_path.exists():
            log.info(f"Loading model weights from: {model_path}")
            _ = model.load_state_dict(load(model_path))
            return model

        log.info("Model weights not found.")
        train_loader, _ = get_cifar10_loaders()

        optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        warmup = LinearLR(
            optimizer,
            start_factor=1 / 32,
            end_factor=32 / 32,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
        scheduler = SequentialLR(optimizer, [warmup, cosine], [warmup_epochs])
        criterion = CrossEntropyLoss()

        start_epoch = 0

        if checkpoint_path.exists():
            checkpoint = load(checkpoint_path, weights_only=True)
            start_epoch = checkpoint["last_epoch"] + 1

            _ = model.load_state_dict(checkpoint["model"])
            _ = optimizer.load_state_dict(checkpoint["optimizer"])
            _ = scheduler.load_state_dict(checkpoint["scheduler"])

            print(f"Resuming from epoch {start_epoch}/{epochs}: {checkpoint_path}")

        for epoch in track(
            sequence=range(start_epoch, epochs),
            description=f"Training for {epochs} epochs ...",
        ):
            _ = model.train()

            inputs: Tensor
            labels: Tensor

            for _inputs_, _labels_ in train_loader:
                inputs, labels = _inputs_.to(torch_device), _labels_.to(torch_device)

                if AT is not None:
                    _ = model.eval()
                    adv_inputs = cast("Tensor", AT(inputs, labels))
                    _ = model.train()

                    inputs = cat([inputs, adv_inputs], dim=0)
                    labels = cat([labels, labels], dim=0)

                elif GDA is not None:
                    inputs = inputs + GDA * randn_like(inputs)

                optimizer.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                optimizer.step()  # pyright: ignore[reportUnknownMemberType, reportUnusedCallResult]

            scheduler.step()

            save(
                {
                    "last_epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                checkpoint_path,
            )

        log.info(f"Saving model to: {model_path}")
        save(model.state_dict(), model_path)
        checkpoint_path.unlink(missing_ok=True)

        return model

    @property
    @cache
    def loaders(self) -> tuple[DataLoaderTensor, DataLoaderTensor]:
        """CIFAR-10 train and test data loaders."""
        return get_cifar10_loaders()
