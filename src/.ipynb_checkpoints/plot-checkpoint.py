import matplotlib.pyplot as plt
from pathlib import Path

from typing import Optional


def plot_losses(
    training_losses, validation_losses, path = Path("loss.pdf"),
):
    """
    Plots the training and validation losses.
    """

    figure = plt.figure(dpi=200, figsize=(8, 6))

    plt.plot(training_losses, label="Training Loss")
    plt.plot(validation_losses, label="Validation Loss")
    plt.legend()
    plt.savefig(path, bbox_inches="tight")


def plot_accuracies(
    training_accuracy, validation_accuracy,  path = Path("accuracy.pdf")):
    """
    Plots the training and validation accuracies.
    """

    figure = plt.figure(dpi=200, figsize=(8, 6))

    plt.plot(training_accuracy, label="Training Accuracy")
    plt.plot(validation_accuracy, label="Validation Accuracy")
    plt.legend()
    plt.savefig(path, bbox_inches="tight")


def plot_line_with_label(
    x, y, label: str, path = Path("line.pdf"),
):
    """
    Plots the training and validation accuracies.
    """

    figure = plt.figure(dpi=200, figsize=(8, 6))

    plt.plot(x, y, label=label)
    plt.legend()
    plt.savefig(path, bbox_inches="tight")


def plot_list_of_lines_and_labels(
    lines_and_labels, log: bool = False, path = Path("all.pdf"),
):
    """
    Plots the training and validation accuracies.
    """

    figure = plt.figure(dpi=200, figsize=(8, 6))

    if len(lines_and_labels[0]) == 2:
        for line, label in lines_and_labels:
            plt.plot(line, label=label)
    elif len(lines_and_labels[0]) == 3:
        for line, label, color in lines_and_labels:
            plt.plot(line, label=label, color=color)

    if log:
        plt.yscale("log")
        plt.xscale("log")

    plt.legend()
    plt.savefig(path, bbox_inches="tight")
