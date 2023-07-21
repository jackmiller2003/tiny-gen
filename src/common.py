import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.model import TinyModel


def get_accuracy_on_dataset(
    model: TinyModel,
    dataset: Dataset,
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


def weight_decay_LP(model, loss, accuracy, lambda_: float, degree: int):
    """
    Computes Lp norm weight decay for the model.

    Args:
        model (nn.Module): PyTorch model
        lambda_ (float): Regularization factor
        degree (int): Degree of polynomial for Lp norm

    Returns:
        float: Lp weight decay
    """
    lp_norm_total = 0.0
    for param in model.parameters():
        if param.requires_grad:
            lp_norm_total += torch.sum(torch.abs(param) ** degree)

    return lambda_ * lp_norm_total


def optimal_set_weight_decay(
    model, loss, accuracy, lambda_: float, gamma_: float, accuracy_cutoff: float
):
    """
    Computes optimal set weight decay
    """
    lp_norm_total = 0.0

    for param in model.parameters():
        if param.requires_grad:
            lp_norm_total += torch.sum(torch.abs(param) ** 2)

    optimal_set_addition = 0.0

    if accuracy > accuracy_cutoff:
        optimal_set_addition = 1.0

    return (lambda_ + gamma_ * optimal_set_addition) * lp_norm_total
