"""Diploma entrypoint file."""

from argparse import ArgumentParser
from dataclasses import dataclass

from .helpers import BaseModel, get_accuracy, log, show_perturbation_example
from .models import Cifar10, Mnist


@dataclass
class MyNamespace:
    """Used for typing the arguments."""

    model: str
    attacks: list[str]
    defence: str | None
    vizualize: bool
    verbose: bool


arg_parser = ArgumentParser()
_ = arg_parser.add_argument(
    "-m",
    "--model",
    required=True,
    choices=["MNIST", "CIFAR10"],
    help="Choose model to use.",
)
_ = arg_parser.add_argument(
    "-a",
    "--attacks",
    required=False,
    choices=["FGSM", "JSMA", "DF", "CW"],
    default=[],
    nargs="*",
    help="Choose attack(s) to perform on the selected `--model`.",
)
_ = arg_parser.add_argument(
    "-d",
    "--defence",
    required=False,
    choices=["AT", "GDA"],
    help="Choose defence to use.",
)
_ = arg_parser.add_argument(
    "-q",
    "--quiet",
    required=False,
    action="store_false",
    dest="vizualize",
    help="Do not display graphical examples.",
)
_ = arg_parser.add_argument(
    "-v",
    "--verbose",
    required=False,
    action="store_true",
    help="Show debug loggin messages.",
)
args = arg_parser.parse_args(namespace=MyNamespace)

models: dict[str, BaseModel] = {
    "CIFAR10": Cifar10(),
    "MNIST": Mnist(),
}

if args.verbose:
    log.setLevel("DEBUG")


GDA_SIGMA = 0.3


def print_formatted_accuracy(
    attack: str | None = None,
    defence: str | None = None,
) -> None:
    """Print accuracy in a fixed-width format."""
    attack = attack or "None"
    defence = defence or "None"

    print(f"Accuracy (Attack: {attack:4}, Defence: {defence:4}): ...", end=" ")


def main() -> None:
    """Execute using `uv run diploma`."""
    model = models[args.model]
    model_clean = model.get()
    test_loader = model.test_loader

    print()
    print_formatted_accuracy(attack=None, defence=None)
    clean_accuracy = get_accuracy(
        model=model_clean,
        dataloader=test_loader,
        attack=None,
        max_samples=None,
        max_batch_size=None,
    )
    print(f"{100 * clean_accuracy:.2f}%")

    for attack_name in args.attacks:
        current_attack, is_slow_attack = model.get_attack(attack_name)

        max_samples: int | None = None
        max_batch_size: int | None = None

        if is_slow_attack:
            max_samples = 500
            max_batch_size = 100

        print()
        print_formatted_accuracy(attack=attack_name, defence=None)
        adversarial_accuracy_undefended = get_accuracy(
            model=model_clean,
            dataloader=test_loader,
            attack=current_attack,
            max_samples=max_samples,
            max_batch_size=max_batch_size,
        )
        print(f"{100 * adversarial_accuracy_undefended:.2f}%")

        if args.vizualize:
            show_perturbation_example(model, current_attack)

        match args.defence:
            case "AT":
                model_defended = model.get(AT=current_attack)
            case "GDA":
                model_defended = model.get(GDA=GDA_SIGMA)
            case _:
                continue

        print_formatted_accuracy(attack=None, defence=args.defence)
        clean_accuracy_defended = get_accuracy(
            model=model_defended,
            dataloader=test_loader,
            attack=None,
            max_samples=None,
            max_batch_size=None,
        )
        print(f"{100 * clean_accuracy_defended:.2f}%")

        print_formatted_accuracy(attack=attack_name, defence=args.defence)
        adversarial_accuracy_defended = get_accuracy(
            model=model_defended,
            dataloader=test_loader,
            attack=current_attack,
            max_samples=max_samples,
            max_batch_size=max_batch_size,
        )
        print(f"{100 * adversarial_accuracy_defended:.2f}%")
