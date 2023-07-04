import torch

from src.dataset import ParityPredictionDataset
from src.model import TinyModel, MyHingeLoss
from src.common import get_accuracy

from tqdm import tqdm


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
) -> tuple[TinyModel, list[float], list[float], list[float], list[float]]:
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
        dataset=training_dataset, batch_size=batch_size, shuffle=True
    )

    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset, batch_size=batch_size, shuffle=True
    )

    if loss_function_label == "hinge":
        loss_function = MyHingeLoss()
    elif loss_function_label == "mse":
        loss_function = torch.nn.MSELoss()
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

    for epoch in tqdm(range(epochs), disable=not progress_bar):
        total_loss = 0
        total_accuracy = 0
        number_batches = 0

        for batch in train_loader:
            inputs = batch[0]
            targets = batch[1]

            predictions = model(inputs)

            loss = loss_function(predictions, targets)

            total_loss += loss.item()
            accuracy = get_accuracy(predictions, targets)

            total_accuracy += accuracy
            number_batches += 1

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        training_losses.append(total_loss / number_batches)
        training_accuracy.append(total_accuracy / number_batches)

        total_val_loss = 0
        total_val_accuracy = 0
        number_val_batches = 0

        # Run validation loss
        with torch.no_grad():
            for batch in validation_loader:
                inputs = batch[0]
                targets = batch[1]

                predictions = model(inputs)

                validation_loss = loss_function(predictions, targets)

                total_val_loss += float(validation_loss.item())
                number_val_batches += 1
                total_val_accuracy += get_accuracy(predictions, targets)

        validation_losses.append(total_val_loss / number_val_batches)
        validation_accuracy.append(total_val_accuracy / number_val_batches)

    return (
        model,
        training_losses,
        validation_losses,
        training_accuracy,
        validation_accuracy,
    )
