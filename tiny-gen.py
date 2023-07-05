import torch
import argparse
import numpy as np

from src.dataset import ParityPredictionDataset
from src.model import TinyModel
from src.train import train_model
from src.plot import (
    plot_losses,
    plot_accuracies,
    plot_line_with_label,
    plot_list_of_lines_and_labels,
)
from src.common import get_accuracy_on_dataset


def experiment_1(args):
    """
    Completes experiment 1.

    I want to see at what point the network is able to generalise beyond
    the training data to complete the task of parity prediction.

    To do this, we start with 3 random seeds and walk through the k ranges:
    * [2,2]
    * [2,3]
    * [2,4]
    * [2,5]

    Then, we look at the test results on the generalisation dataset which goes from
    k=6 to k=10. We will use a sequence length of 10.
    """

    # Replicability
    np.random.seed(0)

    average_accuracies = []

    max_k_factor = 10
    binary_sequence_length = 10

    total_sequence_length = max_k_factor + binary_sequence_length

    generalisation_datasets = []

    # TODO: this method might be biased?
    random_points = np.random.uniform(6, max_k_factor, 1)

    for k_factor_sample in random_points:
        generalisation_datasets.append(
            ParityPredictionDataset(
                num_samples=args.num_samples,
                sequence_length=binary_sequence_length,
                k_factor_range=[int(k_factor_sample), int(k_factor_sample)],
                max_k_factor=max_k_factor,
            )
        )

    for k_range in [(2, 2), (2, 3), (2, 4), (2, 5)]:
        # Create the training dataset
        training_dataset = ParityPredictionDataset(
            num_samples=args.num_samples,
            sequence_length=binary_sequence_length,
            k_factor_range=k_range,
            max_k_factor=max_k_factor,
        )

        average_accuracy = 0

        for random_seed_index in range(0, 3):
            # Set random torch seed
            torch.manual_seed(random_seed_index)

            # Create the model
            model = TinyModel(
                input_size=total_sequence_length,
                hidden_layer_size=args.hidden_layer_size,
                output_size=1,
                random_seed=0,
            )

            # Train the model
            (
                model,
                training_losses,
                validation_losses,
                training_accuracy,
                validation_accuracy,
                _,
            ) = train_model(
                training_dataset=training_dataset,
                validation_dataset=training_dataset,
                model=model,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                batch_size=args.batch_size,
                loss_function_label=args.loss_function_label,
                optimiser_function_label=args.optimiser_label,
                progress_bar=False,
            )

            # Test the model on the generalisation dataset
            accuracy = sum(
                [
                    get_accuracy_on_dataset(model, generalisation_dataset)
                    for generalisation_dataset in generalisation_datasets
                ]
            ) / len(generalisation_datasets)

            average_accuracy += accuracy

            print(
                f"Random seed: {random_seed_index}, k_range: {k_range}, final_val_accuracy: {round(validation_accuracy[-1],2)}, accuracy: {accuracy}"
            )

        average_accuracy /= 3
        average_accuracies.append(average_accuracy)

    print(f"Average accuracies: {average_accuracies}")

    # Plot average accuracies
    # TODO: egh hard coded!
    plot_line_with_label(x=[2, 3, 4, 5], y=average_accuracies, label="Average accuracy")


def experiment_2(args):
    """
    The aim of this experiment is to determine whether we get grokking behaviour.

    That is, train on k=2,3,4,5 and then at each epoch test on k=6. If we were to see
    grokking then we would expect to see the accuracy on k=6 to increase drastically
    at one point.
    """

    num_samples = 1000
    binary_sequence_length = 6
    k_factor_range = [2, 5]
    max_k_factor = 6
    hidden_layer_size = 10000  # Overparameterised regime
    epochs = 300

    total_sequence_length = max_k_factor + binary_sequence_length

    # Create the training dataset
    training_dataset = ParityPredictionDataset(
        num_samples=num_samples,
        sequence_length=binary_sequence_length,
        k_factor_range=k_factor_range,
        max_k_factor=max_k_factor,
    )

    generalisation_dataset = ParityPredictionDataset(
        num_samples=round(num_samples / 10),
        sequence_length=binary_sequence_length,
        k_factor_range=[max_k_factor, max_k_factor],
        max_k_factor=max_k_factor,
    )

    # Create the model
    model = TinyModel(
        input_size=total_sequence_length,
        hidden_layer_size=hidden_layer_size,
        output_size=1,
    )

    # Train the model
    (
        model,
        training_losses,
        validation_losses,
        training_accuracy,
        validation_accuracy,
        generalisation_accuracy,
    ) = train_model(
        training_dataset=training_dataset,
        validation_dataset=training_dataset,
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=epochs,
        batch_size=args.batch_size,
        loss_function_label=args.loss_function_label,
        optimiser_function_label=args.optimiser_label,
        progress_bar=True,
        generalisation_dataset=generalisation_dataset,
    )

    plot_list_of_lines_and_labels(
        lines_and_labels=[
            (training_accuracy, "Training accuracy"),
            (validation_accuracy, "Validation accuracy"),
            (generalisation_accuracy, "Generalisation accuracy"),
        ]
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to generate for the dataset",
    )

    argparser.add_argument(
        "--sequence_length",
        type=int,
        default=30,
        help="Length of the sequences to generate",
    )

    argparser.add_argument(
        "--k_factor_range",
        type=int,
        nargs=2,
        default=[1, 3],
        help="Range of k factors to use before generalisation",
    )

    argparser.add_argument(
        "--generalisation_k_factor",
        type=int,
        nargs=2,
        default=[4, 4],
    )

    argparser.add_argument(
        "--max_k_factor",
        type=int,
        default=20,
        help="Maximum k factor to use for one-hot encoding",
    )

    argparser.add_argument(
        "--hidden_layer_size",
        type=int,
        default=100,
        help="Size of the hidden layer in the network",
    )

    argparser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-1,
        help="Learning rate for the network",
    )

    argparser.add_argument(
        "--weight_decay", type=float, default=1e-2, help="Weight decay for the network"
    )

    argparser.add_argument(
        "--epochs", type=int, default=200, help="Number of epochs to train for"
    )

    argparser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )

    argparser.add_argument(
        "--loss_function_label", type=str, default="hinge", help="Loss function to use"
    )

    argparser.add_argument(
        "--optimiser_label", type=str, default="sgd", help="Optimiser to use"
    )

    argparser.add_argument(
        "--plot_loss", type=bool, default=True, help="Whether to plot the loss"
    )

    argparser.add_argument(
        "--plot_accuracy", type=bool, default=True, help="Whether to plot the accuracy"
    )

    argparser.add_argument(
        "--experiments", type=int, nargs="+", default=[], help="Experiments to run"
    )

    args = argparser.parse_args()

    if 1 in args.experiments:
        experiment_1(args)

    if 2 in args.experiments:
        experiment_2(args)

    if args.experiments != []:
        exit()

    dataset = ParityPredictionDataset(
        num_samples=args.num_samples,
        sequence_length=args.sequence_length,
        k_factor_range=args.k_factor_range,
        max_k_factor=args.max_k_factor,
    )

    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [0.7, 0.2, 0.1]
    )

    input_size = args.sequence_length + args.max_k_factor

    print(f"Using an input size of {input_size}")

    model = TinyModel(
        input_size=input_size,
        hidden_layer_size=args.hidden_layer_size,
        output_size=1,
    )

    (
        model,
        training_losses,
        validation_losses,
        training_accuracy,
        validation_accuracy,
        _,
    ) = train_model(
        training_dataset=train_dataset,
        validation_dataset=validation_dataset,
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        loss_function_label=args.loss_function_label,
        optimiser_function_label=args.optimiser_label,
    )

    if args.plot_loss:
        plot_losses(
            training_losses=training_losses, validation_losses=validation_losses
        )

    print(f"Final training loss: {training_losses[-1]}")
    print(f"Final validation loss: {validation_losses[-1]}")

    if args.plot_accuracy:
        plot_accuracies(
            training_accuracy=training_accuracy, validation_accuracy=validation_accuracy
        )

    print(f"Final training accuracy: {training_accuracy[-1]}")
    print(f"Final validation accuracy: {validation_accuracy[-1]}")

    # Test the model on a new dataset with the generalisation k factor

    accuracies = []

    for k_factor in range(
        args.generalisation_k_factor[0], args.generalisation_k_factor[1] + 1
    ):
        # TODO: this is somewhat unclean. We should probably have a separate dataset for a single k.
        generalisation_dataset = ParityPredictionDataset(
            num_samples=args.num_samples,
            sequence_length=args.sequence_length,
            k_factor_range=[k_factor, k_factor],
            max_k_factor=args.max_k_factor,
        )

        accuracy = get_accuracy_on_dataset(model, generalisation_dataset)
        accuracies.append(accuracy)

    print(f"Generalisation accuracies: {accuracies}")
