import matplotlib.pyplot as plt
from pathlib import Path


def plot_losses(
    training_losses: list[float],
    validation_losses: list[float],
    path: Path = Path("loss.pdf"),
) -> None:
    """
    Plots the training and validation losses.
    """

    figure = plt.figure(dpi=200, figsize=(8, 6))

    plt.plot(training_losses, label="Training Loss")
    plt.plot(validation_losses, label="Validation Loss")
    plt.legend()
    plt.savefig(path, bbox_inches="tight")


def plot_accuracies(
    training_accuracy: list[float],
    validation_accuracy: list[float],
    path: Path = Path("accuracy.pdf"),
) -> None:
    """
    Plots the training and validation accuracies.
    """

    figure = plt.figure(dpi=200, figsize=(8, 6))

    plt.plot(training_accuracy, label="Training Accuracy")
    plt.plot(validation_accuracy, label="Validation Accuracy")
    plt.legend()
    plt.savefig(path, bbox_inches="tight")


def plot_line_with_label(
    x: list[float],
    y: list[float],
    label: str,
    path: Path = Path("line.pdf"),
) -> None:
    """
    Plots the training and validation accuracies.
    """

    figure = plt.figure(dpi=200, figsize=(8, 6))

    plt.plot(x, y, label=label)
    plt.legend()
    plt.savefig(path, bbox_inches="tight")
