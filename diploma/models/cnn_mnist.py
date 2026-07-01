"""CNN model for MNIST classification."""

from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, cast, override

from rich.progress import track
from torch import Tensor, cat, load, randn_like, save
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    CrossEntropyLoss,
    Dropout,
    Flatten,
    Linear,
    MaxPool2d,
    Module,
    ReLU,
    Sequential,
)
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from diploma.datasets import MNIST_CLASSES, MNIST_NORMALIZATION, get_mnist_loaders
from diploma.helpers import BaseModel, DataLoaderTensor, get_torch_device, log

if TYPE_CHECKING:
    from torchattacks.attack import Attack


class Mnist(BaseModel):
    """MNIST classifier model class."""

    CMAP: str | None = "gray"
    """
    The MNIST datasets consists of grayscale samples => a value of `"gray"` is used.
    """

    def __init__(self) -> None:
        """Initialize the MNIST model."""
        super().__init__(
            name="cnn_mnist",
            normalization=MNIST_NORMALIZATION,
        )

    @cache
    def get(
        self,
        epochs: int = 15,
        warmup_epochs: int = 2,
        AT: Attack | None = None,
        GDA: float | None = None,
    ) -> Module:
        """Return CNN model for MNIST classification.

        Using the default values for `epochs` and `warmup_epochs`
        gives around 99% accuracy.

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
                error = f"GDA strength must be in the range [0, 1], got {GDA}."
                raise ValueError(error)

            log.info(f"Using GDA defence with strength σ={GDA} ...")

            model_path = Path(f"./data/weights/{self.name}_GDA.pth")
            checkpoint_path = Path(f"./data/checkpoints/{self.name}_GDA.pth")

        else:
            model_path = Path(f"./data/weights/{self.name}.pth")
            checkpoint_path = Path(f"./data/checkpoints/{self.name}.pth")

        model_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        model = MnistCNN()
        model = model.to(torch_device)

        if model_path.exists():
            log.info(f"Loading model weights from: {model_path}")
            _ = model.load_state_dict(load(model_path))
            return model

        log.info("Model weights not found.")
        train_loader, _ = get_mnist_loaders()

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
        """MNIST train and test data loaders."""
        return get_mnist_loaders()


class MnistCNN(Module):
    """CNN for MNIST."""

    def __init__(self) -> None:
        """Initialize the CNN."""
        super().__init__()

        self.features: Sequential = Sequential(
            Conv2d(1, 32, kernel_size=3, padding=1),
            BatchNorm2d(32),
            ReLU(inplace=True),
            Conv2d(32, 64, kernel_size=3, padding=1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(2),
            Conv2d(64, 128, kernel_size=3, padding=1),
            BatchNorm2d(128),
            ReLU(inplace=True),
            MaxPool2d(2),
        )

        self.classifier: Sequential = Sequential(
            Flatten(),
            Linear(128 * 7 * 7, 256),
            ReLU(inplace=True),
            Dropout(0.5),
            Linear(256, MNIST_CLASSES),
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.features(x))
