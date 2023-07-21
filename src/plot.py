import torch.nn as nn
import torch

import matplotlib.pyplot as plt
from pathlib import Path

from typing import Optional

import numpy.typing as npt
import numpy as np

from copy import deepcopy

from tqdm import tqdm


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


def plot_2D_loss_surface_with_random_directions(
    initial_model: nn.Module,
    trained_model: nn.Module,
    training_dataset: torch.utils.data.Dataset,
    validation_dataset: torch.utils.data.Dataset,
    weight_decay: float,
    models: list = None,
) -> None:
    """
    Plots the loss surface of the model.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()

    trained_model = deepcopy(trained_model)
    trained_model.to(device)

    theta_init = [p.clone().detach() for p in initial_model.parameters()]
    theta_star = [p.clone().detach() for p in trained_model.parameters()]

    training_dataloader = torch.utils.data.DataLoader(
        dataset=training_dataset, batch_size=32, shuffle=False, num_workers=4
    )

    validation_dataloader = torch.utils.data.DataLoader(
        dataset=validation_dataset, batch_size=32, shuffle=False, num_workers=4
    )

    # Generate two random directions
    d1 = [torch.tensor(w1 - w2).to(device) for w1, w2 in zip(theta_star, theta_init)]
    d2 = [torch.randn(w.size()).to(device) for w in theta_star]
    # d1 = [torch.randn(w.size()).to(device) for w in theta_star]

    d1_tensor = torch.cat([d.view(-1) for d in d1])
    d2_tensor = torch.cat([d.view(-1) for d in d2])
    theta_init_tensor = torch.cat([t.view(-1) for t in theta_init])
    theta_star_tensor = torch.cat([t.view(-1) for t in theta_star])

    # Choose x points so that the range includes the initial and final points
    if models is not None:
        loss_surface_points = []
        for model in models:
            model.to(device)

            # Convert model parameters to a tensor
            theta_model_tensor = torch.cat(
                [param.view(-1) for param in model.parameters()]
            )

            # Compute projections
            projection_d1 = torch.tensordot(theta_model_tensor, d1_tensor, dims=1)
            projection_d2 = torch.tensordot(theta_model_tensor, d2_tensor, dims=1)

            loss_surface_points.append(
                np.array([projection_d1.item(), projection_d2.item()])
            )

        print(f"Model projections are: {loss_surface_points}")

        # Project theta_init onto d1 and d2
        theta_init_on_d1 = torch.tensordot(theta_init_tensor, d1_tensor, dims=1)
        theta_init_on_d2 = torch.tensordot(theta_init_tensor, d2_tensor, dims=1)

        theta_final_on_d1 = torch.tensordot(theta_star_tensor, d1_tensor, dims=1)
        theta_final_on_d2 = torch.tensordot(theta_star_tensor, d2_tensor, dims=1)

    loss_surface_points = np.array(loss_surface_points)

    h = 2e2
    n_steps = 10
    loss_surface = np.zeros((n_steps, n_steps))
    validation_surface = np.zeros((n_steps, n_steps))
    weight_decay_surface = np.zeros((n_steps, n_steps))

    # Choose x_points and y_points so that they include the initial and final points
    x_min = min(theta_init_on_d1.item(), theta_final_on_d1.item()) - 5
    x_max = max(theta_init_on_d1.item(), theta_final_on_d1.item()) + 5
    y_min = min(theta_init_on_d2.item(), theta_final_on_d2.item()) - 5
    y_max = max(theta_init_on_d2.item(), theta_final_on_d2.item()) + 5

    x_points = np.linspace(x_min, x_max, n_steps)
    y_points = np.linspace(y_min, y_max, n_steps)

    for i, x_point in tqdm(enumerate(x_points), total=n_steps):
        for j, y_point in enumerate(y_points):
            # Update weights
            for k, param in enumerate(trained_model.parameters()):
                param.data = theta_star[k] + x_point * d1[k] + y_point * d2[k]

            # Compute loss on validation set
            loss = 0
            validation_loss = 0
            weight_decay_loss = 0
            with torch.no_grad():
                for inputs, targets in training_dataloader:
                    inputs = inputs.to(device, non_blocking=False)
                    targets = targets.to(device, non_blocking=False)
                    outputs = trained_model(inputs)
                    l2_norm_of_weights = sum(
                        torch.norm(param) ** 2 for param in trained_model.parameters()
                    )

                    loss += criterion(outputs, targets).item()
                    weight_decay_loss += weight_decay * l2_norm_of_weights

                for inputs, targets in validation_dataloader:
                    inputs = inputs.to(device, non_blocking=False)
                    targets = targets.to(device, non_blocking=False)
                    outputs = trained_model(inputs)
                    validation_loss += criterion(outputs, targets).item()

            print(f"Loss is {loss / len(training_dataloader)}")
            print(f"Validation loss is {validation_loss / len(validation_dataloader)}")
            print(
                f"Weight decay loss is {weight_decay_loss / len(training_dataloader)}"
            )

            loss_surface[i, j] = loss / len(training_dataloader)
            validation_surface[i, j] = validation_loss / len(validation_dataloader)

            weight_decay_surface[i, j] = weight_decay_loss / len(training_dataloader)

    figure = plt.figure(dpi=200, figsize=(10, 10))

    print(f"Weight decay surface: {weight_decay_surface}")
    print(f"Validation surface: {validation_surface}")
    print(f"Loss surface: {loss_surface}")

    # Plot the loss surface
    plt.imshow(
        loss_surface,
        cmap="viridis",
        interpolation="bilinear",
        extent=[x_min, x_max, y_min, y_max],
        aspect="auto",
    )
    plt.colorbar()
    plt.title("Loss Surface")
    plt.xlabel("Training direction")
    plt.ylabel("Direction 2")

    print(loss_surface_points)

    # Plot the points with a line
    if models is not None:
        plt.scatter(loss_surface_points[:, 0], loss_surface_points[:, 1], color="black")
        plt.plot(loss_surface_points[:, 0], loss_surface_points[:, 1], color="red")

        plt.scatter(
            theta_init_on_d1.item(),
            theta_init_on_d2.item(),
            color="black",
            marker="x",
        )

        plt.scatter(
            theta_final_on_d1.item(),
            theta_final_on_d2.item(),
            color="green",
            marker="x",
        )

    plt.savefig("loss_surface-t.png", bbox_inches="tight")

    plt.close()

    figure = plt.figure(dpi=200, figsize=(10, 10))

    plt.imshow(
        validation_surface,
        cmap="viridis",
        interpolation="bilinear",
        extent=[x_min, x_max, y_min, y_max],
        aspect="auto",
    )
    plt.colorbar()
    plt.title("Validaiton Surface")
    plt.xlabel("Training direction")
    plt.ylabel("Direction 2")

    if models is not None:
        plt.scatter(loss_surface_points[:, 0], loss_surface_points[:, 1], color="black")
        plt.plot(loss_surface_points[:, 0], loss_surface_points[:, 1], color="red")

        plt.scatter(
            theta_init_on_d1.item(),
            theta_init_on_d2.item(),
            color="black",
            marker="x",
        )

        plt.scatter(
            theta_final_on_d1.item(),
            theta_final_on_d2.item(),
            color="green",
            marker="x",
        )

    plt.savefig("validation_surface-t.png", bbox_inches="tight")

    plt.close()

    figure = plt.figure(dpi=200, figsize=(10, 10))

    plt.imshow(
        weight_decay_surface,
        cmap="viridis",
        interpolation="bilinear",
        extent=[x_min, x_max, y_min, y_max],
        aspect="auto",
    )
    plt.colorbar()
    plt.title("Weight decay")
    plt.xlabel("Training direction")
    plt.ylabel("Direction 2")

    if models is not None:
        plt.scatter(loss_surface_points[:, 0], loss_surface_points[:, 1], color="black")
        plt.plot(loss_surface_points[:, 0], loss_surface_points[:, 1], color="red")

        plt.scatter(
            theta_init_on_d1.item(),
            theta_init_on_d2.item(),
            color="black",
            marker="x",
        )

        plt.scatter(
            theta_final_on_d1.item(),
            theta_final_on_d2.item(),
            color="green",
            marker="x",
        )

    plt.savefig("weight_decay-t.png", bbox_inches="tight")
