import torch

from src.dataset import ParityPredictionDataset
from src.model import TinyModel, MyHingeLoss
from src.common import get_accuracy, get_accuracy_on_dataset
from src.plot import plot_heatmap

from tqdm import tqdm
import os
from pathlib import Path


def train_model(
    training_dataset: ParityPredictionDataset,
    validation_dataset: ParityPredictionDataset,
    model: TinyModel,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    loss_function_label: str,
    optimiser_function_label: str,
    progress_bar: bool = True,
    weight_matrix_path: Path = None,
    generalisation_dataset: ParityPredictionDataset = None,
    rate_limit: list[tuple] = None,
) -> tuple[TinyModel, list[float], list[float], list[float], list[float], list[float]]:
    """
    Function for training TinyModel.

    Returns
        - model: TinyModel
        - training_losses: list[float]
        - validation_losses: list[float]
        - training_accuracy: list[float]
        - validation_accuracy: list[float]
    """

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

    # Check what device the loader is on
    # print(f"Training loader on device: {next(iter(train_loader))[0].device}")

    if loss_function_label == "hinge":
        loss_function = MyHingeLoss()
    elif loss_function_label == "mse":
        loss_function = torch.nn.MSELoss()
    elif loss_function_label == "cross-entropy":
        loss_function = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("Invalid loss function.")

    if optimiser_function_label == "sgd":
        optimiser = torch.optim.SGD(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        raise ValueError("Invalid optimiser.")

    training_losses, training_accuracy = [], []
    validation_losses, validation_accuracy = [], []

    generalisation_accuracy = []

    for epoch in tqdm(
        range(epochs),
        disable=not progress_bar,
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    ):
        total_loss = 0
        total_accuracy = 0
        number_batches = 0

        if rate_limit is not None:
            # print(f"rate limit {rate_limit}")
            for layer, frequency in rate_limit:
                if epoch % frequency == 0:
                    model.unfreeze([layer])
                    print(
                        f"Unfreezing layer {layer}, rate limit {frequency}, epoch {epoch}"
                    )
                else:
                    model.freeze([layer])
                    print(
                        f"Freezing layer {layer}, rate limit {frequency}, epoch {epoch}"
                    )

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

        # time_after_training = torch.cuda.Event(enable_timing=True)
        # time_after_training.record()

        # print(f"Epoch {epoch} with training took {time_before_training.elapsed_time(time_after_training)}ms")

        training_losses.append(total_loss / number_batches)
        training_accuracy.append(total_accuracy / number_batches)

        total_val_loss = 0
        total_val_accuracy = 0
        number_val_batches = 0

        # time_before_validation = torch.cuda.Event(enable_timing=True)
        # time_before_validation.record()

        # Run validation loss
        with torch.no_grad():
            for batch in validation_loader:
                inputs = batch[0].to(device, non_blocking=False)
                targets = batch[1].to(device, non_blocking=False)

                predictions = model(inputs)

                validation_loss = loss_function(predictions, targets)

                total_val_loss += float(validation_loss.item())
                number_val_batches += 1
                total_val_accuracy += get_accuracy(predictions, targets)

        # time_after_validation = torch.cuda.Event(enable_timing=True)
        # time_after_validation.record()

        # print(f"Epoch {epoch} with validation took {time_before_validation.elapsed_time(time_after_validation)}ms")

        validation_losses.append(total_val_loss / number_val_batches)
        validation_accuracy.append(total_val_accuracy / number_val_batches)

        # time_before_generalisation = torch.cuda.Event(enable_timing=True)
        # time_before_generalisation.record()

        if generalisation_dataset is not None:
            generalisation_accuracy.append(
                get_accuracy_on_dataset(model, generalisation_dataset)
            )

        # TODO: egh, the path addition is not using the library. Will also crash out if there aren't three layers!
        if weight_matrix_path is not None and epoch % 10 == 0:
            os.makedirs(f"{weight_matrix_path}/{epoch}", exist_ok=True)
            weights_layer_1 = model.look(1)
            weights_layer_2 = model.look(2)
            # weights_layer_3 = model.look(3)

            plot_heatmap(
                weights_layer_1,
                path=Path(f"{weight_matrix_path}/{epoch}/weights_layer_1.png"),
            )

            plot_heatmap(
                weights_layer_2,
                path=Path(f"{weight_matrix_path}/{epoch}/weights_layer_2.png"),
            )

            # plot_heatmap(
            #     weights_layer_3,
            #     path=Path(f"{weight_matrix_path}/{epoch}/weights_layer_3.png"),
            # )

        # time_after_generalisation = torch.cuda.Event(enable_timing=True)
        # time_after_generalisation.record()

        # print(f"Epoch {epoch} with generalisation took {time_before_generalisation.elapsed_time(time_after_generalisation)}ms")

    return (
        model,
        training_losses,
        validation_losses,
        training_accuracy,
        validation_accuracy,
        generalisation_accuracy,
    )
