import torch

from src.dataset import ParityPredictionDataset
from src.model import TinyModel


def get_accuracy_on_dataset(
    model: TinyModel,
    dataset: ParityPredictionDataset,
    batch_size: int = 32,
    verbose: bool = False,
) -> float:
    """
    Gets the accuracy on a dataset given a particular trained model.
    """

    total_accuracy = 0

    test_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False
    )

    number_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch

            predictions = model(inputs)

            total_accuracy += get_accuracy(predictions, targets, verbose=verbose)
            number_batches += 1

    return total_accuracy / number_batches


def get_accuracy(
    predictions: torch.Tensor, targets: torch.Tensor, verbose: bool = False
) -> float:
    """
    Computes the accuracy of the predictions given the targets.
    """

    if verbose:
        print(f"predictions: {torch.squeeze(predictions)}")
        print(f"targets: {targets}")

    return float(
        torch.mean((torch.sign(torch.squeeze(predictions)) == targets).float())
    )
