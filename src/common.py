import torch
import torch.nn.functional as F

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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            predictions = model(inputs)

            total_accuracy += get_accuracy(predictions, targets, verbose=verbose)
            number_batches += 1

    return total_accuracy / number_batches


# TODO: this is horrible. Fix it.
def get_accuracy(
    predictions: torch.Tensor, targets: torch.Tensor, verbose: bool = False
) -> float:
    """
    Computes the accuracy of the predictions given the targets.
    """

    if verbose:
        print(f"predictions: {torch.squeeze(predictions)}")
        print(f"targets: {targets.shape}")

    # Check the size of targets. If there is only one target per example, then
    # return current. Otherwise, do a one-hot comparison with 0.5 as correct.

    if len(targets.shape) == 1:
        return float(
            torch.mean((torch.sign(torch.squeeze(predictions)) == targets).float())
        )
    else:
        predictions = F.softmax(predictions, dim=1)

        return float(
            torch.mean(
                (
                    torch.argmax(predictions, dim=1) == torch.argmax(targets, dim=1)
                ).float()
            )
        )
