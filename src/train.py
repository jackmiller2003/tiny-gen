import torch
from torch.utils.data import Dataset
import torch.nn as nn

from src.model import TinyModel, MyHingeLoss
from src.common import get_accuracy, get_accuracy_on_dataset
from src.plot import plot_list_of_lines_and_labels, plot_heatmap

from tqdm import tqdm
import os
from pathlib import Path

import numpy as np
import numpy.typing as npt


class Observer:
    """
    Class for watching model training which is initialised with relevant components to record
    different parts of training.
    """

    def __init__(
        self,
        observation_settings: dict = {},
        generalisation_datasets: dict[str, Dataset] = {},
    ):
        self.observation_settings = observation_settings
        self.generalisation_datasets = generalisation_datasets

        self.training_losses = []
        self.validation_losses = []
        self.training_accuracy = []
        self.validation_accuracy = []

        self.weight_norms = {}
        self.weights = {}
        self.generalisation_score = {}

        for name in self.generalisation_datasets.keys():
            self.generalisation_score[name] = []

        for observation, setting in self.observation_settings.items():
            if observation == "weight_norm":
                layers = setting["layers"]
                for layer in layers:
                    self.weight_norms[layer] = []
            elif observation == "weights":
                layers = setting["layers"]
                for layer in layers:
                    self.weights[layer] = []

    def record_training_loss(self, loss: float) -> None:
        self.training_losses.append(loss)

    def record_validation_loss(self, loss: float) -> None:
        self.validation_losses.append(loss)

    def record_training_accuracy(self, accuracy: float) -> None:
        self.training_accuracy.append(accuracy)

    def record_validation_accuracy(self, accuracy: float) -> None:
        self.validation_accuracy.append(accuracy)

    def observe_weights(self, model: nn.Module) -> None:
        """
        Observes various components of the weights based on specifications in
        the observations dictionary.
        """

        # The model must have a look method which maps from layer number to weights
        assert hasattr(model, "look")

        for observation, setting in self.observation_settings.items():
            if observation == "weight_norm":
                layers = setting["layers"]
                norms = [np.linalg.norm(model.look(layer), p=2) for layer in layers]
                [
                    self.weight_norms[layer].append(norm)
                    for layer, norm in zip(layers, norms)
                ]
            elif observation == "weights":
                layers = setting["layers"]
                [self.weights[layer].append(model.look(layer)) for layer in layers]

    def observe_generalisation(self, model: nn.Module) -> None:
        for name, dataset in self.generalisation_datasets.items():
            self.generalisation_score[name].append(
                get_accuracy_on_dataset(model, dataset)
            )

    def plot_me(
        self, path: Path, file_extension: str = ".png", log: bool = True
    ) -> None:
        """
        Plots all of the observations the class has recorded.
        """

        os.makedirs(path / Path("weights"), exist_ok=True)

        # --- Validation and training information --- #

        plot_list_of_lines_and_labels(
            [
                (self.training_losses, "Training Loss"),
                (self.validation_losses, "Validation Loss"),
            ],
            path=path / Path("loss" + file_extension),
            log=log,
        )

        plot_list_of_lines_and_labels(
            [
                (self.training_accuracy, "Training Accuracy"),
                (self.validation_accuracy, "Validation Accuracy"),
            ],
            path=path / Path("accuracy" + file_extension),
            log=log,
        )

        # --- Weight information --- #

        plot_list_of_lines_and_labels(
            [
                (norm_list, layer_number)
                for layer_number, norm_list in self.weight_norms.items()
            ],
            path=path / Path("weight_norm" + file_extension),
            log=log,
        )

        if "weights" in self.observation_settings:
            weight_matrix_frequency = self.observation_settings["weights"]["frequency"]

        for layer_number, weight_list in self.weights.items():
            for weight_number, weight_matrix in enumerate(weight_list):
                if weight_number % weight_matrix_frequency == 0:
                    plot_heatmap(
                        weight_matrix,
                        path=path
                        / Path(
                            f"weights/{layer_number}_{weight_number}" + file_extension
                        ),
                    )

        # --- Generalisation information --- #

        plot_list_of_lines_and_labels(
            [
                (generalisation_accuracy, name)
                for name, generalisation_accuracy in self.generalisation_score.items()
            ],
            path=path / Path("generalisation" + file_extension),
            log=log,
        )


def train_model(
    training_dataset: Dataset,
    validation_dataset: Dataset,
    model: TinyModel,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    loss_function_label: str,
    optimiser_function_label: str,
    progress_bar: bool = True,
    rate_limit: list[tuple] = None,
    observer: Observer = None,
) -> tuple[nn.Module, Observer]:
    if observer is None:
        observer = Observer()

    train_loader = torch.utils.data.DataLoader(
        dataset=training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if loss_function_label == "hinge":
        print("Using hinge loss")
        loss_function = MyHingeLoss()
    elif loss_function_label == "mse":
        print("Using mse loss")
        loss_function = torch.nn.MSELoss()
    elif loss_function_label == "cross-entropy":
        print("Using cross-entropy loss")
        loss_function = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("Invalid loss function.")

    if optimiser_function_label == "sgd":
        optimiser = torch.optim.SGD(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        raise ValueError("Invalid optimiser.")

    for epoch in tqdm(
        range(epochs),
        disable=not progress_bar,
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    ):
        total_loss = total_accuracy = number_batches = 0

        if rate_limit is not None:
            for layer, frequency in rate_limit:
                if epoch % frequency == 0:
                    model.unfreeze([layer])
                else:
                    model.freeze([layer])

        for batch in train_loader:
            inputs = batch[0].contiguous().to(device, non_blocking=False)
            targets = batch[1].contiguous().to(device, non_blocking=False)

            predictions = model(inputs)

            loss = loss_function(predictions, targets)

            total_loss += loss.item()
            accuracy = get_accuracy(predictions, targets)

            total_accuracy += accuracy
            number_batches += 1

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        observer.record_training_loss(total_loss / number_batches)
        observer.record_training_accuracy(total_accuracy / number_batches)

        total_val_loss = total_val_accuracy = number_val_batches = 0

        with torch.no_grad():
            for batch in validation_loader:
                inputs = batch[0].to(device, non_blocking=False)
                targets = batch[1].to(device, non_blocking=False)

                predictions = model(inputs)

                validation_loss = loss_function(predictions, targets)

                total_val_loss += float(validation_loss.item())
                number_val_batches += 1
                total_val_accuracy += get_accuracy(predictions, targets)

        observer.record_validation_loss(total_val_loss / number_val_batches)
        observer.record_validation_accuracy(total_val_accuracy / number_val_batches)

        observer.observe_generalisation(model)
        observer.observe_weights(model)

    return (model, observer)
