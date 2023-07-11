"""
This file contains a list of experiments. The general paradigm to follow when writing
an experiment is to:
    1. Describe the experiment in the docstring
    2. Write the code to run the experiment
    3. Write the code to plot the results of the experiment

Somebody else using the code should be able to run the experiment by running the
relevant function in this file via the flag --experiment.

For reproducibility define a random seed or set of random seeds, passing them into:
    1. The dataset
    2. The model
Theoretically, one should be able to get exactly the same results as the author
by running the experiment with the same random seeds.
"""

import torch
from torch.utils.data import TensorDataset
import argparse
import numpy as np
from pathlib import Path
import os

from src.dataset import ParityTask, HiddenDataset, PeekParityTask
from src.model import TinyModel, ExpandableModel
from src.train import train_model, Observer
from src.plot import (
    plot_losses,
    plot_accuracies,
    plot_line_with_label,
    plot_list_of_lines_and_labels,
    plot_validation_and_accuracy_from_observers,
)
from src.common import get_accuracy_on_dataset


def experiment_grokking_plain(args):
    """
    Can we recover grokking behaviour using cross-entropy loss
    from the hidden parity prediction paper:

    https://arxiv.org/abs/2303.11873.
    """

    weight_decay = 1e-2
    learning_rate = 1e-1
    batch_size = 32
    input_size = 40
    ouput_size = 2
    k = 3
    hidden_size = 1000
    epochs = 2
    number_training_samples = 1000
    number_validation_samples = 100
    random_seed = 0

    # Create the training dataset
    parity_task_dataset = ParityTask(
        sequence_length=k,
        num_samples=number_training_samples + number_validation_samples,
        random_seed=random_seed,
    )

    hidden_parity_task_dataset = HiddenDataset(
        dataset=parity_task_dataset,
        total_length=input_size,
        random_seed=random_seed,
    )

    training_dataset, validation_dataset = torch.utils.data.random_split(
        hidden_parity_task_dataset,
        [number_training_samples, number_validation_samples],
    )

    model = TinyModel(
        input_size=input_size,
        hidden_layer_size=hidden_size,
        output_size=ouput_size,
        random_seed=random_seed,
    )

    (model, observer) = train_model(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        batch_size=batch_size,
        loss_function_label="cross-entropy",
        optimiser_function_label="sgd",
        progress_bar=True,
    )

    observer.plot_me(path=Path("experiments/grokking_plain/"))


def experiment_data_sensitvity(args):
    """
    Here we are going to look at the sensitivity of grokking to the size of the underlying dataset
    for the parity prediction task. We will use the same model as in experiment 0.
    """

    weight_decay = 1e-2
    learning_rate = 1e-1
    batch_size = 32
    input_size = 40
    ouput_size = 2
    k = 3
    hidden_size = 1000
    epochs = 2
    number_validation_samples = 100
    random_seed = 0

    training_dataset_sizes = [550, 770, 1100]

    observers = []

    for training_dataset_size in training_dataset_sizes:
        entire_dataset = ParityTask(
            sequence_length=k,
            num_samples=training_dataset_size + number_validation_samples,
            random_seed=random_seed,
        )

        hidden_dataset = HiddenDataset(
            dataset=entire_dataset,
            total_length=input_size,
            random_seed=random_seed,
        )

        training_dataset, validation_dataset = torch.utils.data.random_split(
            hidden_dataset,
            [training_dataset_size, number_validation_samples],
        )

        model = TinyModel(
            input_size=input_size,
            hidden_layer_size=hidden_size,
            output_size=ouput_size,
            random_seed=random_seed,
        )

        (model, observer) = train_model(
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            loss_function_label="cross-entropy",
            optimiser_function_label="sgd",
            progress_bar=True,
        )

        observers.append(observer)

    plot_validation_and_accuracy_from_observers(
        observers=observers,
        label_list=training_dataset_sizes,
        path=Path("experiments/data_sensitvity/"),
    )


def experiment_random_features(args):
    """
    Does grokking occur under random feature regression? We freeze the first layer
    weights under several random seeds to find out.
    """

    weight_decay = 1e-2
    learning_rate = 1e-1
    batch_size = 32
    input_size = 40
    ouput_size = 2
    k = 3
    hidden_size = 1000
    epochs = 2
    number_training_samples = 1000
    number_validation_samples = 100

    random_seeds = [0, 1, 2]

    observers = []

    for random_seed in random_seeds:
        entire_dataset = ParityTask(
            sequence_length=k,
            num_samples=number_training_samples + number_validation_samples,
            random_seed=random_seed,
        )

        hidden_dataset = HiddenDataset(
            dataset=entire_dataset,
            total_length=input_size,
            random_seed=random_seed,
        )

        training_dataset, validation_dataset = torch.utils.data.random_split(
            hidden_dataset,
            [number_training_samples, number_validation_samples],
        )

        model = TinyModel(
            input_size=input_size,
            hidden_layer_size=hidden_size,
            output_size=ouput_size,
            random_seed=random_seed,
        )

        (model, observer) = train_model(
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            loss_function_label="cross-entropy",
            optimiser_function_label="sgd",
            progress_bar=True,
        )

        observers.append(observer)

    plot_validation_and_accuracy_from_observers(
        observers=observers,
        label_list=random_seeds,
        path=Path("experiments/random_features/"),
    )


def experiment_two_layer(args):
    """
    Does Grokking occur for a two layer network?
    """

    weight_decay = 1e-2
    learning_rate = 1e-1
    batch_size = 32
    input_size = 40
    ouput_size = 2
    k = 3
    hidden_size_1 = 1000
    hidden_size_2 = 1000
    epochs = 20
    number_training_samples = 1000
    number_validation_samples = 100
    random_seed = 0

    entire_dataset = ParityTask(
        sequence_length=k,
        num_samples=number_training_samples + number_validation_samples,
        random_seed=random_seed,
    )

    hidden_dataset = HiddenDataset(
        dataset=entire_dataset,
        total_length=input_size,
        random_seed=random_seed,
    )

    training_dataset, validation_dataset = torch.utils.data.random_split(
        hidden_dataset,
        [number_training_samples, number_validation_samples],
    )

    model = ExpandableModel(
        input_size=input_size,
        hidden_layer_sizes=[hidden_size_1, hidden_size_2],
        output_size=ouput_size,
        random_seed=random_seed,
    )

    observer = Observer(
        observation_settings={"weights": {"frequency": 10, "layers": [1, 2]}}
    )

    (model, observer) = train_model(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        batch_size=batch_size,
        loss_function_label="cross-entropy",
        optimiser_function_label="sgd",
        progress_bar=True,
        observer=observer,
    )

    observer.plot_me(path=Path("experiments/two_layer/"))


def experiment_random_feature_3_layer(args):
    """
    In this experiment, I wonder if something like this will emerge:

    sequence -> random feature -> clear randomness -> grokking pattern -> output

    With a 3-layer network.
    """

    weight_decay = 1e-2
    learning_rate = 1e-1
    batch_size = 32
    input_size = 40
    ouput_size = 2
    k = 3
    hidden_size_1 = 200
    hidden_size_2 = 200
    hidden_size_3 = 200
    epochs = 2
    number_training_samples = 1000
    number_validation_samples = 100
    random_seed = 0

    entire_dataset = ParityTask(
        sequence_length=k,
        num_samples=number_training_samples + number_validation_samples,
        random_seed=random_seed,
    )

    hidden_dataset = HiddenDataset(
        dataset=entire_dataset,
        total_length=input_size,
        random_seed=random_seed,
    )

    training_dataset, validation_dataset = torch.utils.data.random_split(
        hidden_dataset,
        [number_training_samples, number_validation_samples],
    )

    model = ExpandableModel(
        input_size=input_size,
        hidden_layer_sizes=[hidden_size_1, hidden_size_2, hidden_size_3],
        output_size=ouput_size,
        random_seed=random_seed,
    )

    model.freeze([1])

    observer = Observer(
        observation_settings={"weights": {"frequency": 10, "layers": [2]}}
    )

    (model, observer) = train_model(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        batch_size=batch_size,
        loss_function_label="cross-entropy",
        optimiser_function_label="sgd",
        progress_bar=True,
        observer=observer,
    )

    observer.plot_me(path=Path("experiments/random_feature_3_layer/"))


def experiment_double_grokking(args):
    """
    This experiment is designed to see if we can uncover a double grokking scenario. That is, the network shifts from:
    confusion -> generalisation to some of the pattern -> full pattern
    """

    weight_decay = 1e-2
    learning_rate = 1e-1
    batch_size = 32
    input_size = 40
    ouput_size = 2
    k = 3
    hidden_size = 1000
    epochs = 200
    number_training_samples = 1000
    number_validation_samples = 400
    random_seed = 0

    entire_dataset = PeekParityTask(
        sequence_length=k,
        num_samples=number_training_samples + number_validation_samples,
        random_seed=random_seed,
        peek_condition=[1, 1, -1],
    )

    hidden_dataset = HiddenDataset(
        dataset=entire_dataset,
        total_length=input_size,
        random_seed=random_seed,
    )

    training_dataset, validation_dataset = torch.utils.data.random_split(
        hidden_dataset,
        [number_training_samples, number_validation_samples],
    )

    indices_of_second_task = [
        i
        for i, x in enumerate(validation_dataset)
        if (x[0][:3] == torch.tensor([1, 1, 0])).all()
    ]

    print(f"Found {len(indices_of_second_task)} samples in the second task.")

    generalisation_dataset = torch.utils.data.Subset(
        validation_dataset, indices_of_second_task
    )

    model = TinyModel(
        input_size=input_size,
        hidden_layer_size=hidden_size,
        output_size=ouput_size,
        random_seed=random_seed,
    )

    observer = Observer(
        observation_settings={"weights": {"frequency": 10, "layers": [1]}},
        generalisation_datasets={"peek_condition": generalisation_dataset},
    )

    (model, observer) = train_model(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        batch_size=batch_size,
        loss_function_label="cross-entropy",
        optimiser_function_label="sgd",
        progress_bar=True,
        observer=observer,
    )

    observer.plot_me(path=Path("experiments/double_grokking/"), log=False)


def experiment_rate_limiting(args):
    """
    There exists a hypothesis that grokking is a competition between different
    parts of the network. To test this we will rate limit different components
    of the network by freezing the weights at certain frequencies.
    """

    rate_limiting_tuples = [
        [(1, 64), (1, 1)],
        [(1, 32), (1, 1)],
        [(1, 16), (1, 1)],
        [(1, 8), (1, 1)],
        [(1, 4), (1, 1)],
        [(1, 2), (1, 1)],
        [(1, 1), (2, 1)],
        [(1, 1), (2, 2)],
        [(1, 1), (2, 4)],
        [(1, 1), (2, 8)],
        [(1, 1), (2, 16)],
        [(1, 1), (2, 32)],
        [(1, 1), (2, 64)],
    ]

    weight_decay = 1e-2
    learning_rate = 1e-1
    batch_size = 32
    input_size = 40
    ouput_size = 2
    k = 3
    hidden_size = 1000
    epochs = 200
    number_training_samples = 1000
    number_validation_samples = 400
    random_seed = 0

    entire_dataset = ParityTask(
        sequence_length=k,
        num_samples=number_training_samples + number_validation_samples,
        random_seed=random_seed,
    )

    hidden_dataset = HiddenDataset(
        dataset=entire_dataset,
        total_length=input_size,
        random_seed=random_seed,
    )

    training_dataset, validation_dataset = torch.utils.data.random_split(
        hidden_dataset,
        [number_training_samples, number_validation_samples],
    )

    for rate_limit in rate_limiting_tuples:
        # Create the model
        model = TinyModel(
            input_size=input_size,
            hidden_layer_size=hidden_size,
            output_size=ouput_size,
            random_seed=random_seed,
        )

        print(f"Testing rate limit {rate_limit}")
        (
            model,
            observer,
        ) = train_model(
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            loss_function_label="cross-entropy",
            optimiser_function_label="sgd",
            rate_limit=rate_limit,
            progress_bar=True,
        )

        observer.plot_me(
            path=Path(
                f"experiments/rate_limiting/accuracy_{rate_limit[0]}_{rate_limit[1]}"
            ),
            log=False,
        )


# def experiment_11(args):
#     """
#     Can we reproduce grokking within modula arithmetic?
#     """

#     p = 4
#     num_samples = 770
#     epochs = 50
#     loss_function_label = "cross-entropy"
#     hidden_layer_size = 200

#     training_dataset = ModularArithmeticTask(num_samples=num_samples, p=p)

#     number_training_samples = int(num_samples * 0.90909) + 1
#     number_validation_samples = num_samples - number_training_samples

#     training_dataset, validation_dataset = torch.utils.data.random_split(
#         training_dataset,
#         [number_training_samples, number_validation_samples],
#     )

#     print(f"Training dataset size: {len(training_dataset)}")
#     print(f"Validation dataset size: {len(validation_dataset)}")

#     # Create the model
#     model = TinyModel(
#         input_size=2 * p, hidden_layer_size=hidden_layer_size, output_size=p
#     )

#     # Train the model
#     (
#         model,
#         training_losses,
#         validation_losses,
#         training_accuracy,
#         validation_accuracy,
#         generalisation_accuracy,
#     ) = train_model(
#         training_dataset=training_dataset,
#         validation_dataset=validation_dataset,
#         model=model,
#         learning_rate=args.learning_rate,
#         weight_decay=args.weight_decay,
#         epochs=epochs,
#         batch_size=args.batch_size,
#         loss_function_label=loss_function_label,
#         optimiser_function_label=args.optimiser_label,
#         weight_matrix_path=Path("experiments/experiment_11/weights"),
#     )

#     plot_list_of_lines_and_labels(
#         lines_and_labels=[
#             (training_losses, "Training loss"),
#             (validation_losses, "Validation loss"),
#         ],
#         log=False,
#         path=Path(f"experiments/experiment_11/loss.png"),
#     )

#     plot_list_of_lines_and_labels(
#         lines_and_labels=[
#             (training_accuracy, "Training accuracy"),
#             (validation_accuracy, "Validation accuracy"),
#         ],
#         log=False,
#         path=Path(f"experiments/experiment_11/accuracy.png"),
#     )


# def experiment_12(args):
#     """
#     Can we reproduce grokking within modula arithmetic?
#     """

#     p = 4
#     total_sequnce_size = 40
#     num_samples = 770
#     epochs = 150
#     loss_function_label = "cross-entropy"
#     hidden_layer_size = 300

#     training_dataset = HiddenModularArithmeticTask(
#         num_samples=num_samples, p=p, sequence_length=total_sequnce_size
#     )

#     number_training_samples = int(num_samples * 0.90909) + 1
#     number_validation_samples = num_samples - number_training_samples

#     training_dataset, validation_dataset = torch.utils.data.random_split(
#         training_dataset,
#         [number_training_samples, number_validation_samples],
#     )

#     print(f"Training dataset size: {len(training_dataset)}")
#     print(f"Validation dataset size: {len(validation_dataset)}")

#     # Create the model
#     model = TinyModel(
#         input_size=total_sequnce_size,
#         hidden_layer_size=hidden_layer_size,
#         output_size=p,
#     )

#     # Train the model
#     (
#         model,
#         training_losses,
#         validation_losses,
#         training_accuracy,
#         validation_accuracy,
#         generalisation_accuracy,
#     ) = train_model(
#         training_dataset=training_dataset,
#         validation_dataset=validation_dataset,
#         model=model,
#         learning_rate=args.learning_rate,
#         weight_decay=args.weight_decay,
#         epochs=epochs,
#         batch_size=args.batch_size,
#         loss_function_label=loss_function_label,
#         optimiser_function_label=args.optimiser_label,
#         weight_matrix_path=Path("experiments/experiment_12/weights"),
#     )

#     plot_list_of_lines_and_labels(
#         lines_and_labels=[
#             (training_losses, "Training loss"),
#             (validation_losses, "Validation loss"),
#         ],
#         log=False,
#         path=Path(f"experiments/experiment_12/loss.png"),
#     )

#     plot_list_of_lines_and_labels(
#         lines_and_labels=[
#             (training_accuracy, "Training accuracy"),
#             (validation_accuracy, "Validation accuracy"),
#         ],
#         log=False,
#         path=Path(f"experiments/experiment_12/accuracy.png"),
#     )


# def experiment_13(args):
#     """
#     Making sure we see grokking on the parity prediction task with cross entropy.
#     """

#     weight_decay = 1e-2
#     learning_rate = 1e-1
#     batch_size = 32
#     hidden_size = 200
#     number_samples = 1100
#     epochs = 200

#     # Create the training dataset
#     entire_dataset = HiddenParityPrediction(
#         num_samples=number_samples, sequence_length=40, k=3, for_cross_entropy=True
#     )

#     # Split into training and validation should be 1000 and 100
#     training_dataset, validation_dataset = torch.utils.data.random_split(
#         entire_dataset,
#         [int(number_samples * 0.90909) + 1, int(number_samples * 0.09091)],
#     )

#     print(f"Training dataset size: {len(training_dataset)}")
#     print(f"Validation dataset size: {len(validation_dataset)}")

#     # Create the model
#     model = TinyModel(
#         input_size=40,
#         hidden_layer_size=hidden_size,
#         output_size=2,
#         random_seed=0,
#     )

#     # Train the model
#     (
#         model,
#         training_losses,
#         validation_losses,
#         training_accuracy,
#         validation_accuracy,
#         _,
#     ) = train_model(
#         training_dataset=training_dataset,
#         validation_dataset=validation_dataset,
#         model=model,
#         learning_rate=learning_rate,
#         weight_decay=weight_decay,
#         epochs=epochs,
#         batch_size=batch_size,
#         loss_function_label="cross-entropy",
#         optimiser_function_label="sgd",
#         progress_bar=True,
#         weight_matrix_path=Path("experiments/experiment_13/weights"),
#     )

#     plot_list_of_lines_and_labels(
#         lines_and_labels=[
#             (training_accuracy, "Training accuracy"),
#             (validation_accuracy, "Validation accuracy"),
#         ],
#         log=True,
#         path=Path("experiments/experiment_13/accuracy.png"),
#     )

#     plot_list_of_lines_and_labels(
#         lines_and_labels=[
#             (training_losses, "Training loss"),
#             (validation_losses, "Validation loss"),
#         ],
#         log=True,
#         path=Path("experiments/experiment_13/loss.png"),
#     )


# def experiment_14(args):
#     """
#     Combined prediction task.
#     """

#     prime_dataset = HiddenModularArithmeticTask(
#         num_samples=550, p=4, sequence_length=40
#     )

#     parity_dataset = HiddenParityPrediction(
#         num_samples=1100, sequence_length=40, k=3, for_cross_entropy=True
#     )

#     # Theoretically the output should require at most 2^8 + 2^3 = 256 + 8 = 264 bits.

#     combined_dataset = combine_datasets(prime_dataset, parity_dataset)

#     training_dataset, validation_dataset = torch.utils.data.random_split(
#         combined_dataset,
#         [int(1650 * 0.90909) + 1, int(1650 * 0.09091)],
#     )

#     print(f"Training dataset size: {len(training_dataset)}")
#     print(f"Validation dataset size: {len(validation_dataset)}")

#     # Create the model
#     model = TinyModel(input_size=80, hidden_layer_size=264, output_size=4 + 2)

#     # Train the model
#     (
#         model,
#         training_losses,
#         validation_losses,
#         training_accuracy,
#         validation_accuracy,
#         _,
#     ) = train_model(
#         training_dataset=training_dataset,
#         validation_dataset=validation_dataset,
#         model=model,
#         learning_rate=1e-1,
#         weight_decay=1e-2,
#         epochs=400,
#         batch_size=32,
#         loss_function_label="cross-entropy",
#         optimiser_function_label="sgd",
#         progress_bar=True,
#         weight_matrix_path=Path("experiments/experiment_14/weights"),
#     )

#     plot_list_of_lines_and_labels(
#         lines_and_labels=[
#             (training_accuracy, "Training accuracy"),
#             (validation_accuracy, "Validation accuracy"),
#         ],
#         log=True,
#         path=Path("experiments/experiment_14/accuracy.png"),
#     )

#     plot_list_of_lines_and_labels(
#         lines_and_labels=[
#             (training_losses, "Training loss"),
#             (validation_losses, "Validation loss"),
#         ],
#         log=True,
#         path=Path("experiments/experiment_14/loss.png"),
#     )


# def experiment_15(args):
#     """
#     Pretty much a repeat of above but we continue to constrain the number of layers...
#     """

#     prime_dataset = HiddenModularArithmeticTask(
#         num_samples=550, p=4, sequence_length=40
#     )

#     parity_dataset = HiddenParityPrediction(
#         num_samples=1100, sequence_length=40, k=3, for_cross_entropy=True
#     )

#     combined_dataset = combine_datasets(prime_dataset, parity_dataset)

#     training_dataset, validation_dataset = torch.utils.data.random_split(
#         combined_dataset,
#         [int(1650 * 0.90909) + 1, int(1650 * 0.09091)],
#     )

#     print(f"Training dataset size: {len(training_dataset)}")
#     print(f"Validation dataset size: {len(validation_dataset)}")

#     hidden_layer_sizes = [264, 256, 128, 64, 32]

#     training_accuracies = []
#     validation_accuracies = []
#     training_losses_all = []
#     validation_losses_all = []

#     for hidden_layer_size in hidden_layer_sizes:
#         # Create the model
#         model = TinyModel(
#             input_size=80, hidden_layer_size=hidden_layer_size, output_size=4 + 2
#         )

#         # Train the model
#         (
#             model,
#             training_losses,
#             validation_losses,
#             training_accuracy,
#             validation_accuracy,
#             _,
#         ) = train_model(
#             training_dataset=training_dataset,
#             validation_dataset=validation_dataset,
#             model=model,
#             learning_rate=1e-1,
#             weight_decay=1e-2,
#             epochs=200,  # change to 200 epochs
#             batch_size=32,
#             loss_function_label="cross-entropy",
#             optimiser_function_label="sgd",
#             progress_bar=True,
#             weight_matrix_path=Path(
#                 f"experiments/experiment_15/weights/{hidden_layer_size}"
#             ),
#         )

#         # Append the accuracy and losses to the lists
#         training_accuracies.append(training_accuracy)
#         validation_accuracies.append(validation_accuracy)
#         training_losses_all.append(training_losses)
#         validation_losses_all.append(validation_losses)

#     # Plot accuracy
#     lines_and_labels = [
#         (acc, f"Training accuracy {size}")
#         for acc, size in zip(training_accuracies, hidden_layer_sizes)
#     ]
#     lines_and_labels += [
#         (acc, f"Validation accuracy {size}")
#         for acc, size in zip(validation_accuracies, hidden_layer_sizes)
#     ]
#     plot_list_of_lines_and_labels(
#         lines_and_labels=lines_and_labels,
#         log=True,
#         path=Path("experiments/experiment_15/accuracy.png"),
#     )

#     # Plot loss
#     lines_and_labels = [
#         (loss, f"Training loss {size}")
#         for loss, size in zip(training_losses_all, hidden_layer_sizes)
#     ]
#     lines_and_labels += [
#         (loss, f"Validation loss {size}")
#         for loss, size in zip(validation_losses_all, hidden_layer_sizes)
#     ]
#     plot_list_of_lines_and_labels(
#         lines_and_labels=lines_and_labels,
#         log=True,
#         path=Path("experiments/experiment_15/loss.png"),
#     )


# def experiment_16(args):
#     """
#     Looking at dependence of grokking on the random feature length.
#     """

#     weight_decay = 1e-2
#     learning_rate = 1e-1
#     batch_size = 32
#     hidden_size = 200
#     number_samples = 770
#     epochs = 100

#     training_loss_list = []
#     validation_loss_list = []
#     training_accuracies = []
#     validation_accuracies = []

#     sequence_lengths = [6, 10, 20, 40]

#     for sequence_length in sequence_lengths:
#         # Create the training dataset
#         entire_dataset = HiddenParityPrediction(
#             num_samples=number_samples,
#             sequence_length=sequence_length,
#             k=3,
#             for_cross_entropy=True,
#         )

#         training, validation = torch.utils.data.random_split(
#             entire_dataset,
#             [int(number_samples * 0.90909) + 1, int(number_samples * 0.09091)],
#         )

#         model = TinyModel(
#             input_size=sequence_length,
#             hidden_layer_size=hidden_size,
#             output_size=2,
#             random_seed=0,
#         )

#         # Train model
#         (
#             model,
#             training_losses,
#             validation_losses,
#             training_accuracy,
#             validation_accuracy,
#             _,
#         ) = train_model(
#             training_dataset=training,
#             validation_dataset=validation,
#             model=model,
#             learning_rate=learning_rate,
#             weight_decay=weight_decay,
#             epochs=epochs,
#             batch_size=batch_size,
#             loss_function_label="cross-entropy",
#             optimiser_function_label="sgd",
#             progress_bar=True,
#         )

#         training_accuracies.append(training_accuracy)
#         validation_accuracies.append(validation_accuracy)
#         training_loss_list.append(training_losses)
#         training_loss_list.append(validation_losses)

#     # Plot accuracy
#     lines_and_labels = [
#         (acc, f"Training accuracy {size}")
#         for acc, size in zip(training_accuracies, sequence_lengths)
#     ]

#     lines_and_labels.extend(
#         [
#             (acc, f"Validation accuracy {size}")
#             for acc, size in zip(validation_accuracies, sequence_lengths)
#         ]
#     )

#     plot_list_of_lines_and_labels(
#         lines_and_labels=lines_and_labels,
#         log=True,
#         path=Path("experiments/experiment_16/accuracy.png"),
#     )

#     # Plot loss
#     lines_and_labels = [
#         (loss, f"Training loss {size}")
#         for loss, size in zip(training_loss_list, sequence_lengths)
#     ]

#     lines_and_labels.extend(
#         [
#             (loss, f"Validation loss {size}")
#             for loss, size in zip(validation_loss_list, sequence_lengths)
#         ]
#     )

#     plot_list_of_lines_and_labels(
#         lines_and_labels=lines_and_labels,
#         log=True,
#         path=Path("experiments/experiment_16/loss.png"),
#     )


# def experiment_17(args):
#     """
#     Decrease the weight norm to get rid of grokking.
#     """

#     number_samples = 1100
#     sequence_length = 40
#     hidden_size = 200

#     entire_dataset = HiddenParityPrediction(
#         num_samples=number_samples,
#         sequence_length=sequence_length,
#         k=3,
#         for_cross_entropy=True,
#     )

#     training, validation = torch.utils.data.random_split(
#         entire_dataset,
#         [int(number_samples * 0.90909) + 1, int(number_samples * 0.09091)],
#     )

#     weight_norm_sizes = [1e-8, 1e-6]  # 1e-5, 1e-3, 1e-1, 1, 1e1]

#     training_accuracies = []
#     validation_accuracies = []

#     for weight_norm_size in weight_norm_sizes:
#         model = TinyModel(
#             input_size=sequence_length,
#             hidden_layer_size=hidden_size,
#             output_size=2,
#             random_seed=0,
#         )

#         # Multiply the weights by the weight norm size
#         for param in model.parameters():
#             param.data *= weight_norm_size

#         (
#             model,
#             training_losses,
#             validation_losses,
#             training_accuracy,
#             validation_accuracy,
#             _,
#         ) = train_model(
#             training_dataset=training,
#             validation_dataset=validation,
#             model=model,
#             learning_rate=1e-1,
#             weight_decay=1e-2,
#             epochs=500,
#             batch_size=32,
#             progress_bar=True,
#             loss_function_label="cross-entropy",
#             optimiser_function_label="sgd",
#         )

#         training_accuracies.append(training_accuracy)
#         validation_accuracies.append(validation_accuracy)

#     # Plot accuracy
#     lines_and_labels = [
#         (acc, f"Training accuracy {size}")
#         for acc, size in zip(training_accuracies, weight_norm_sizes)
#     ]

#     lines_and_labels.extend(
#         [
#             (acc, f"Validation accuracy {size}")
#             for acc, size in zip(validation_accuracies, weight_norm_sizes)
#         ]
#     )

#     plot_list_of_lines_and_labels(
#         lines_and_labels=lines_and_labels,
#         log=True,
#         path=Path("experiments/experiment_17/accuracy-small.png"),
#     )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--num_samples",
        type=int,
        default=5500,
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
        default=1000,
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
        "--epochs", type=int, default=500, help="Number of epochs to train for"
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
        "--experiments", type=str, nargs="+", default=[], help="Experiments to run"
    )

    args = argparser.parse_args()

    for i in args.experiments:
        eval("experiment_{}(args)".format(i))

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
