from __future__ import annotations

import matplotlib.pyplot as plt
from pathlib import Path

from typing import Optional

import numpy.typing as npt
import numpy as np


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


def plot_list_of_lines_and_labels(
    lines_and_labels: list[tuple[list[float], str, Optional[str], Optional[str]]],
    log: bool = False,
    path: Path = Path("all.pdf"),
) -> None:
    """
    Plots the training and validation accuracies.
    """

    if lines_and_labels == []:
        return

    figure = plt.figure(dpi=200, figsize=(10, 6))

    if len(lines_and_labels[0]) == 2:
        for line, label in lines_and_labels:
            plt.plot(line, label=label)
    elif len(lines_and_labels[0]) == 3:
        for line, label, color in lines_and_labels:
            plt.plot(line, label=label, color=color)
    elif len(lines_and_labels[0]) == 4:
        for line, label, color, line_type in lines_and_labels:
            plt.plot(line, label=label, color=color, linestyle=line_type)

    if log:
        plt.yscale("log")
        plt.xscale("log")

    # Legend at location bottom right
    plt.legend(loc="lower right")

    plt.savefig(path, bbox_inches="tight")


def plot_heatmap(
    data: npt.NDArray[np.float64],
    path: Path = Path("experiments/experiment_4/heatmap.png"),
) -> None:
    """
    Plots the training and validation accuracies.
    """

    figure = plt.figure(dpi=200, figsize=(10, 6))

    plt.imshow(data, cmap="hot", interpolation="nearest", aspect="auto")
    plt.colorbar()
    plt.savefig(path, bbox_inches="tight")

    plt.close()


def plot_validation_and_accuracy_from_observers(
    observers: list[object], label_list: list, path: Path
) -> None:
    assert len(observers) == len(label_list)
    assert len(observers) <= 5

    colors = ["red", "blue", "green", "orange", "purple"]

    loss_lists_and_labels = []
    accuracy_lists_and_labels = []

    for dataset_size, observer in zip(label_list, observers):
        color = colors.pop()

        loss_lists_and_labels.append(
            (observer.training_losses, f"Training loss {dataset_size}", color, "solid")
        )

        loss_lists_and_labels.append(
            (
                observer.validation_losses,
                f"Validation loss {dataset_size}",
                color,
                "dashed",
            )
        )

        accuracy_lists_and_labels.append(
            (
                observer.training_accuracy,
                f"Training accuracy {dataset_size}",
                color,
                "solid",
            )
        )

        accuracy_lists_and_labels.append(
            (
                observer.validation_accuracy,
                f"Validation accuracy {dataset_size}",
                color,
                "dashed",
            )
        )

    plot_list_of_lines_and_labels(
        lines_and_labels=loss_lists_and_labels,
        log=True,
        path=path / Path("loss.png"),
    )

    plot_list_of_lines_and_labels(
        lines_and_labels=accuracy_lists_and_labels,
        log=True,
        path=path / Path("accuracy.png"),
    )
