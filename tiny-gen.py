import torch
import argparse
import numpy as np
from pathlib import Path
import os

from src.dataset import (
    ParityPredictionDataset,
    HiddenParityPrediction,
    HiddenPeekParityPrediction,
    ModularArithmeticTask,
    HiddenModularArithmeticTask,
    combine_datasets,
)
from src.model import TinyModel, BigModel, ExpandableModel
from src.train import train_model
from src.plot import (
    plot_losses,
    plot_accuracies,
    plot_line_with_label,
    plot_list_of_lines_and_labels,
    plot_heatmap,
)
from src.common import get_accuracy_on_dataset


def experiment_0(args):
    """
    Can we recover grokking from the original paper with an unkown k?

    Yes we can!
    """

    weight_decay = 1e-2
    learning_rate = 1e-1
    batch_size = 32
    hidden_size = 200
    number_samples = 1500
    epochs = 200

    # Replicability
    np.random.seed(0)

    # Create the training dataset
    entire_dataset = HiddenParityPrediction(
        num_samples=number_samples,
        sequence_length=40,
        k=3,
    )

    # Split into training and validation should be 1000 and 100
    training_dataset, validation_dataset = torch.utils.data.random_split(
        entire_dataset,
        [int(number_samples * 0.90909) + 1, int(number_samples * 0.09091)],
    )

    print(f"Training dataset size: {len(training_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")

    # Create the model
    model = TinyModel(
        input_size=40,
        hidden_layer_size=hidden_size,
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
        validation_dataset=validation_dataset,
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        batch_size=batch_size,
        loss_function_label="hinge",
        optimiser_function_label="sgd",
        progress_bar=True,
        weight_matrix_path=Path("experiments/experiment_0/weights"),
    )

    plot_list_of_lines_and_labels(
        lines_and_labels=[
            (training_accuracy, "Training accuracy"),
            (validation_accuracy, "Validation accuracy"),
        ],
        log=True,
        path=Path("experiments/experiment_0/accuracy.png"),
    )

    plot_list_of_lines_and_labels(
        lines_and_labels=[
            (training_losses, "Training loss"),
            (validation_losses, "Validation loss"),
        ],
        log=True,
        path=Path("experiments/experiment_0/loss.png"),
    )


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
    plot_line_with_label(
        x=[2, 3, 4, 5],
        y=average_accuracies,
        label="Average accuracy",
        path=Path("experiments/experiment_1/average_accuracy.png"),
    )


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
    hidden_layer_size = 1000  # Overparameterised regime
    epochs = 300

    total_sequence_length = max_k_factor + binary_sequence_length

    # Create the training dataset
    entire_dataset = ParityPredictionDataset(
        num_samples=num_samples,
        sequence_length=binary_sequence_length,
        k_factor_range=k_factor_range,
        max_k_factor=max_k_factor,
    )

    # Split into training and validation
    training_dataset, validation_dataset = torch.utils.data.random_split(
        entire_dataset, [int(num_samples * 0.8), int(num_samples * 0.2)]
    )

    print(f"Training dataset size: {len(training_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")

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
        validation_dataset=validation_dataset,
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


def experiment_3(args):
    """
    Here we are going to look at the sensitivity of grokking to the size of the underlying dataset.
    """

    dataset_sizes = [220, 550, 770, 1100]
    binary_sequence_length = 40
    epochs = 300

    # List of tuples of the form (training_accuracy, validation_accuracy, training_loss, validation_loss)
    list_of_results = []

    for dataset_size in dataset_sizes:
        entire_dataset = HiddenParityPrediction(
            num_samples=dataset_size,
            sequence_length=binary_sequence_length,
            k=3,
        )

        # Split into training and validation should be 1000 and 100
        training_dataset, validation_dataset = torch.utils.data.random_split(
            entire_dataset,
            [int(dataset_size * 0.90909) + 1, int(dataset_size * 0.09091)],
        )

        print(f"Training dataset size: {len(training_dataset)}")
        print(f"Validation dataset size: {len(validation_dataset)}")

        # Create the model
        model = TinyModel(
            input_size=40,
            hidden_layer_size=1000,
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
            validation_dataset=validation_dataset,
            model=model,
            learning_rate=1e-1,
            weight_decay=1e-2,
            epochs=int(300 * (1000 / dataset_size)),
            batch_size=32,
            loss_function_label="hinge",
            optimiser_function_label="sgd",
            progress_bar=True,
        )

        list_of_results.append(
            (
                training_accuracy,
                validation_accuracy,
                training_losses,
                validation_losses,
            )
        )

    lines_and_labels = []

    colors = ["red", "blue", "green", "orange", "purple"]

    for index, dataset_size in enumerate(dataset_sizes):
        lines_and_labels.append(
            (
                list_of_results[index][0],
                f"Training accuracy, dataset size {dataset_size}",
                colors[index],
            )
        )
        lines_and_labels.append(
            (
                list_of_results[index][1],
                f"Validation accuracy, dataset size {dataset_size}",
                colors[index],
            )
        )

    print(lines_and_labels)

    plot_list_of_lines_and_labels(
        lines_and_labels=lines_and_labels,
        log=True,
        path=Path(f"experiments/experiment_3/accuracy_{dataset_size}.png"),
    )


def experiment_4(args, retrain=False):
    """
    Look inside. Let's have a look at the first layer weights of the model,
    saving them as a heatplot.
    """

    if os.path.exists("models/experiment_4.pt") and not retrain:
        # Load the model
        model = torch.load("models/experiment_4.pt")

    else:
        # Trains a basic model
        dataset = HiddenParityPrediction(num_samples=1100, sequence_length=40, k=3)

        train_dataset, validation_dataset = torch.utils.data.random_split(
            dataset, [1000, 100]
        )

        model = TinyModel(
            input_size=40,
            hidden_layer_size=200,
            output_size=1,
            random_seed=0,
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
            learning_rate=1e-1,
            weight_decay=1e-2,
            epochs=200,
            batch_size=32,
            loss_function_label="hinge",
            optimiser_function_label="sgd",
            progress_bar=True,
        )

        print(f"Final training accuracy: {training_accuracy[-1]}")
        print(f"Final validation accuracy: {validation_accuracy[-1]}")

        # Save the model
        torch.save(model, "models/experiment_4.pt")

    # Plot the weights
    weights_of_layer_2 = model.look(2)
    plot_heatmap(
        weights_of_layer_2, path=Path("experiments/experiment_4/heatmap-2.png")
    )


def experiment_5(args):
    """
    This experiment involves `transplate-initialisation`. What happens when take a learned dense subnetwork
    and transplant partials components of it into a new network? Does Grokking still occur? Likely not.
    """

    # Create a new model
    raise NotImplementedError


def experiment_6(args):
    """
    Does grokking occur under random feature regression? We would guess not right but lets freeze the
    first set of weights and see.

    We begin by using the same arguments as provided in the cli
    """

    training_dataset = HiddenParityPrediction(
        num_samples=args.num_samples,
        sequence_length=args.sequence_length,
        k=3,
    )

    number_training_samples = int(args.num_samples * 0.90909) + 1
    number_validation_samples = args.num_samples - number_training_samples

    # Split into training and validation should be 1000 and 100
    training_dataset, validation_dataset = torch.utils.data.random_split(
        training_dataset,
        [number_training_samples, number_validation_samples],
    )

    print(f"Training dataset size: {len(training_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")

    # Create the model
    model = TinyModel(
        input_size=args.sequence_length,
        hidden_layer_size=args.hidden_layer_size,
        output_size=1,
        random_seed=0,
    )

    # Freeze the first set of weights
    model.freeze([1])

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
        validation_dataset=validation_dataset,
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        loss_function_label=args.loss_function_label,
        optimiser_function_label=args.optimiser_label,
        progress_bar=True,
    )

    plot_list_of_lines_and_labels(
        lines_and_labels=[
            (training_accuracy, "Training accuracy"),
            (validation_accuracy, "Validation accuracy"),
        ],
        log=True,
        path=Path(
            "/scratch/kx58/jm0124/programs/tiny-gen/experiments/experiment_6/accuracy-long.png"
        ),
    )

    plot_list_of_lines_and_labels(
        lines_and_labels=[
            (training_losses, "Training loss"),
            (validation_losses, "Validation loss"),
        ],
        log=True,
        path=Path(
            "/scratch/kx58/jm0124/programs/tiny-gen/experiments/experiment_6/loss-long.png"
        ),
    )


def experiment_7(args):
    """
    In experiment 7 we look at the behaviour of the weights with a two-layer network and whether grokking still
    occurs.
    """

    hidden_layer_1 = 100
    hidden_layer_2 = 10

    training_dataset = HiddenParityPrediction(
        num_samples=args.num_samples,
        sequence_length=args.sequence_length,
        k=3,
    )

    number_training_samples = int(args.num_samples * 0.90909) + 1
    number_validation_samples = args.num_samples - number_training_samples

    # Split into training and validation should be 1000 and 100
    training_dataset, validation_dataset = torch.utils.data.random_split(
        training_dataset,
        [number_training_samples, number_validation_samples],
    )

    print(f"Training dataset size: {len(training_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")

    # Create the model
    model = BigModel(
        input_size=args.sequence_length,
        hidden_layer_sizes=[hidden_layer_1, hidden_layer_2],
        output_size=1,
        random_seed=0,
    )

    model.freeze([1])

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
        validation_dataset=validation_dataset,
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        loss_function_label=args.loss_function_label,
        optimiser_function_label=args.optimiser_label,
        progress_bar=True,
        weight_matrix_path=Path("experiments/experiment_7/weights"),
    )

    plot_list_of_lines_and_labels(
        lines_and_labels=[
            (training_accuracy, "Training accuracy"),
            (validation_accuracy, "Validation accuracy"),
        ],
        log=True,
        path=Path("experiments/experiment_7/accuracy.png"),
    )

    plot_list_of_lines_and_labels(
        lines_and_labels=[
            (training_losses, "Training loss"),
            (validation_losses, "Validation loss"),
        ],
        log=True,
        path=Path("experiments/experiment_7/loss.png"),
    )


def experiment_8(args):
    """
    It seemed that in experiment 7, the internal representation wasn't as clear as in previous grokking scenarios, likely due to
    the presence of the random noise. In this experiment, I wonder if something like this will emerge:

    sequence -> random feature -> clear randomness -> grokking pattern -> output

    With a 3-layer network.
    """

    hidden_layer_size_1 = 100
    hidden_layer_size_2 = 100
    hidden_layer_size_3 = 100

    training_dataset = HiddenParityPrediction(
        num_samples=args.num_samples,
        sequence_length=args.sequence_length,
        k=3,
    )

    number_training_samples = int(args.num_samples * 0.90909) + 1
    number_validation_samples = args.num_samples - number_training_samples

    # Split into training and validation should be 1000 and 100
    training_dataset, validation_dataset = torch.utils.data.random_split(
        training_dataset,
        [number_training_samples, number_validation_samples],
    )

    print(f"Training dataset size: {len(training_dataset)}")

    # Create the model
    model = ExpandableModel(
        input_size=args.sequence_length,
        hidden_layer_sizes=[
            hidden_layer_size_1,
            hidden_layer_size_2,
            hidden_layer_size_3,
        ],
        output_size=1,
    )

    model.freeze([1])

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
        validation_dataset=validation_dataset,
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        loss_function_label=args.loss_function_label,
        optimiser_function_label=args.optimiser_label,
        weight_matrix_path=Path("experiments/experiment_8/weights"),
    )

    plot_list_of_lines_and_labels(
        lines_and_labels=[
            (training_accuracy, "Training accuracy"),
            (validation_accuracy, "Validation accuracy"),
        ],
        log=True,
        path=Path("experiments/experiment_8/accuracy.png"),
    )

    plot_list_of_lines_and_labels(
        lines_and_labels=[
            (training_losses, "Training loss"),
            (validation_losses, "Validation loss"),
        ],
        log=True,
        path=Path("experiments/experiment_8/loss.png"),
    )


def experiment_9(args):
    """
    This experiment is designed to see if we can uncover a double grokking scenario. That is, the network shifts from:
    confusion -> generalisation to some of the pattern -> full pattern

    The dataset is the HiddenPeekParityPrediction which is instantiated in this case with the following behaviour.
    If [1,1,-1] appears then the parity is calcualted with the first 4 elements of the sequence. Otherwise it is not.
    """

    hidden_layer_size = 200
    k = 4
    peek_condition = [1, 1, 1, -1]

    training_dataset = HiddenPeekParityPrediction(
        num_samples=args.num_samples,
        sequence_length=args.sequence_length,
        peek_condition=peek_condition,
        k=k,
    )

    number_training_samples = int(args.num_samples * 0.4) + 1
    number_validation_samples = args.num_samples - number_training_samples

    # Split into training and validation should be 1000 and 100
    training_dataset, validation_dataset = torch.utils.data.random_split(
        training_dataset,
        [number_training_samples, number_validation_samples],
    )

    indices_of_second_task = [
        i
        for i, x in enumerate(validation_dataset)
        if (x[0][:k] == torch.tensor(peek_condition)).all()
    ]

    generalisation_dataset = torch.utils.data.Subset(
        validation_dataset, indices_of_second_task
    )

    print(f"Generalisation dataset size: {len(generalisation_dataset)}")

    print(f"Training dataset size: {len(training_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")

    # Create the model
    model = TinyModel(
        input_size=args.sequence_length,
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
        validation_dataset=validation_dataset,
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        loss_function_label=args.loss_function_label,
        optimiser_function_label=args.optimiser_label,
        weight_matrix_path=Path("experiments/experiment_9/weights"),
        generalisation_dataset=generalisation_dataset,
    )

    plot_list_of_lines_and_labels(
        lines_and_labels=[
            (training_accuracy, "Training accuracy"),
            (validation_accuracy, "Validation accuracy"),
            (generalisation_accuracy, "Generalisation accuracy"),
        ],
        log=True,
        path=Path("experiments/experiment_9/accuracy.png"),
    )

    plot_list_of_lines_and_labels(
        lines_and_labels=[
            (training_losses, "Training loss"),
            (validation_losses, "Validation loss"),
        ],
        log=True,
        path=Path("experiments/experiment_9/loss.png"),
    )


def experiment_10(args):
    """
    Grokking as competition between different parts of the network. To test this we will rate limit different components of the network
    by freezing the weights at certain frequencies.
    """

    rate_limiting_tuples = [
        [(1, 64), (1, 1)],
        [(1, 32), (1, 1)],
        # [(1, 8), (1, 1)],
        # [(1, 4), (1, 1)],
        # [(1, 2), (1, 1)],
        # [(1, 1), (2, 1)],
        # [(1, 1), (2, 2)],
        # [(1, 1), (2, 4)],
        # [(1, 1), (2, 8)],
        # [(1, 1), (2, 16)],
    ]

    entire_dataset = HiddenParityPrediction(
        num_samples=args.num_samples,
        sequence_length=40,
        k=3,
    )

    training_dataset, validation_dataset = torch.utils.data.random_split(
        entire_dataset,
        [
            int(args.num_samples * 0.90909) + 1,
            int(args.num_samples * 0.09091),
        ],
    )

    print(f"Training dataset size: {len(training_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")

    for rate_limit in rate_limiting_tuples:
        # Create the model
        model = TinyModel(
            input_size=40,
            hidden_layer_size=200,
            output_size=1,
            random_seed=0,
        )

        print(f"Testing rate limit {rate_limit}")
        (
            model,
            training_losses,
            validation_losses,
            training_accuracy,
            validation_accuracy,
            generalisation_accuracy,
        ) = train_model(
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            model=model,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            batch_size=args.batch_size,
            loss_function_label=args.loss_function_label,
            optimiser_function_label=args.optimiser_label,
            rate_limit=rate_limit,
        )

        plot_list_of_lines_and_labels(
            lines_and_labels=[
                (training_accuracy, "Training accuracy"),
                (validation_accuracy, "Validation accuracy"),
            ],
            log=True,
            path=Path(
                f"experiments/experiment_10/accuracy_{rate_limit[0]}_{rate_limit[1]}.png"
            ),
        )


def experiment_11(args):
    """
    Can we reproduce grokking within modula arithmetic?
    """

    p = 4
    num_samples = 770
    epochs = 50
    loss_function_label = "cross-entropy"
    hidden_layer_size = 200

    training_dataset = ModularArithmeticTask(num_samples=num_samples, p=p)

    number_training_samples = int(num_samples * 0.90909) + 1
    number_validation_samples = num_samples - number_training_samples

    training_dataset, validation_dataset = torch.utils.data.random_split(
        training_dataset,
        [number_training_samples, number_validation_samples],
    )

    print(f"Training dataset size: {len(training_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")

    # Create the model
    model = TinyModel(
        input_size=2 * p, hidden_layer_size=hidden_layer_size, output_size=p
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
        validation_dataset=validation_dataset,
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=epochs,
        batch_size=args.batch_size,
        loss_function_label=loss_function_label,
        optimiser_function_label=args.optimiser_label,
        weight_matrix_path=Path("experiments/experiment_11/weights"),
    )

    plot_list_of_lines_and_labels(
        lines_and_labels=[
            (training_losses, "Training loss"),
            (validation_losses, "Validation loss"),
        ],
        log=False,
        path=Path(f"experiments/experiment_11/loss.png"),
    )

    plot_list_of_lines_and_labels(
        lines_and_labels=[
            (training_accuracy, "Training accuracy"),
            (validation_accuracy, "Validation accuracy"),
        ],
        log=False,
        path=Path(f"experiments/experiment_11/accuracy.png"),
    )


def experiment_12(args):
    """
    Can we reproduce grokking within modula arithmetic?
    """

    p = 4
    total_sequnce_size = 40
    num_samples = 770
    epochs = 150
    loss_function_label = "cross-entropy"
    hidden_layer_size = 300

    training_dataset = HiddenModularArithmeticTask(
        num_samples=num_samples, p=p, sequence_length=total_sequnce_size
    )

    number_training_samples = int(num_samples * 0.90909) + 1
    number_validation_samples = num_samples - number_training_samples

    training_dataset, validation_dataset = torch.utils.data.random_split(
        training_dataset,
        [number_training_samples, number_validation_samples],
    )

    print(f"Training dataset size: {len(training_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")

    # Create the model
    model = TinyModel(
        input_size=total_sequnce_size,
        hidden_layer_size=hidden_layer_size,
        output_size=p,
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
        validation_dataset=validation_dataset,
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=epochs,
        batch_size=args.batch_size,
        loss_function_label=loss_function_label,
        optimiser_function_label=args.optimiser_label,
        weight_matrix_path=Path("experiments/experiment_12/weights"),
    )

    plot_list_of_lines_and_labels(
        lines_and_labels=[
            (training_losses, "Training loss"),
            (validation_losses, "Validation loss"),
        ],
        log=False,
        path=Path(f"experiments/experiment_12/loss.png"),
    )

    plot_list_of_lines_and_labels(
        lines_and_labels=[
            (training_accuracy, "Training accuracy"),
            (validation_accuracy, "Validation accuracy"),
        ],
        log=False,
        path=Path(f"experiments/experiment_12/accuracy.png"),
    )


def experiment_13(args):
    """
    Making sure we see grokking on the parity prediction task with cross entropy.
    """

    weight_decay = 1e-2
    learning_rate = 1e-1
    batch_size = 32
    hidden_size = 200
    number_samples = 1100
    epochs = 200

    # Create the training dataset
    entire_dataset = HiddenParityPrediction(
        num_samples=number_samples, sequence_length=40, k=3, for_cross_entropy=True
    )

    # Split into training and validation should be 1000 and 100
    training_dataset, validation_dataset = torch.utils.data.random_split(
        entire_dataset,
        [int(number_samples * 0.90909) + 1, int(number_samples * 0.09091)],
    )

    print(f"Training dataset size: {len(training_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")

    # Create the model
    model = TinyModel(
        input_size=40,
        hidden_layer_size=hidden_size,
        output_size=2,
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
        validation_dataset=validation_dataset,
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        batch_size=batch_size,
        loss_function_label="cross-entropy",
        optimiser_function_label="sgd",
        progress_bar=True,
        weight_matrix_path=Path("experiments/experiment_13/weights"),
    )

    plot_list_of_lines_and_labels(
        lines_and_labels=[
            (training_accuracy, "Training accuracy"),
            (validation_accuracy, "Validation accuracy"),
        ],
        log=True,
        path=Path("experiments/experiment_13/accuracy.png"),
    )

    plot_list_of_lines_and_labels(
        lines_and_labels=[
            (training_losses, "Training loss"),
            (validation_losses, "Validation loss"),
        ],
        log=True,
        path=Path("experiments/experiment_13/loss.png"),
    )


def experiment_14(args):
    """
    Combined prediction task.
    """

    prime_dataset = HiddenModularArithmeticTask(
        num_samples=550, p=4, sequence_length=40
    )

    parity_dataset = HiddenParityPrediction(
        num_samples=1100, sequence_length=40, k=3, for_cross_entropy=True
    )

    # Theoretically the output should require at most 2^8 + 2^3 = 256 + 8 = 264 bits.

    combined_dataset = combine_datasets(prime_dataset, parity_dataset)

    training_dataset, validation_dataset = torch.utils.data.random_split(
        combined_dataset,
        [int(1650 * 0.90909) + 1, int(1650 * 0.09091)],
    )

    print(f"Training dataset size: {len(training_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")

    # Create the model
    model = TinyModel(input_size=80, hidden_layer_size=264, output_size=4 + 2)

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
        validation_dataset=validation_dataset,
        model=model,
        learning_rate=1e-1,
        weight_decay=1e-2,
        epochs=400,
        batch_size=32,
        loss_function_label="cross-entropy",
        optimiser_function_label="sgd",
        progress_bar=True,
        weight_matrix_path=Path("experiments/experiment_14/weights"),
    )

    plot_list_of_lines_and_labels(
        lines_and_labels=[
            (training_accuracy, "Training accuracy"),
            (validation_accuracy, "Validation accuracy"),
        ],
        log=True,
        path=Path("experiments/experiment_14/accuracy.png"),
    )

    plot_list_of_lines_and_labels(
        lines_and_labels=[
            (training_losses, "Training loss"),
            (validation_losses, "Validation loss"),
        ],
        log=True,
        path=Path("experiments/experiment_14/loss.png"),
    )


def experiment_15(args):
    """
    Pretty much a repeat of above but we continue to constrain the number of layers...
    """

    prime_dataset = HiddenModularArithmeticTask(
        num_samples=550, p=4, sequence_length=40
    )

    parity_dataset = HiddenParityPrediction(
        num_samples=1100, sequence_length=40, k=3, for_cross_entropy=True
    )

    combined_dataset = combine_datasets(prime_dataset, parity_dataset)

    training_dataset, validation_dataset = torch.utils.data.random_split(
        combined_dataset,
        [int(1650 * 0.90909) + 1, int(1650 * 0.09091)],
    )

    print(f"Training dataset size: {len(training_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")

    hidden_layer_sizes = [264, 256, 128, 64, 32]

    training_accuracies = []
    validation_accuracies = []
    training_losses_all = []
    validation_losses_all = []

    for hidden_layer_size in hidden_layer_sizes:
        # Create the model
        model = TinyModel(
            input_size=80, hidden_layer_size=hidden_layer_size, output_size=4 + 2
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
            validation_dataset=validation_dataset,
            model=model,
            learning_rate=1e-1,
            weight_decay=1e-2,
            epochs=200,  # change to 200 epochs
            batch_size=32,
            loss_function_label="cross-entropy",
            optimiser_function_label="sgd",
            progress_bar=True,
            weight_matrix_path=Path(
                f"experiments/experiment_15/weights/{hidden_layer_size}"
            ),
        )

        # Append the accuracy and losses to the lists
        training_accuracies.append(training_accuracy)
        validation_accuracies.append(validation_accuracy)
        training_losses_all.append(training_losses)
        validation_losses_all.append(validation_losses)

    # Plot accuracy
    lines_and_labels = [
        (acc, f"Training accuracy {size}")
        for acc, size in zip(training_accuracies, hidden_layer_sizes)
    ]
    lines_and_labels += [
        (acc, f"Validation accuracy {size}")
        for acc, size in zip(validation_accuracies, hidden_layer_sizes)
    ]
    plot_list_of_lines_and_labels(
        lines_and_labels=lines_and_labels,
        log=True,
        path=Path("experiments/experiment_15/accuracy.png"),
    )

    # Plot loss
    lines_and_labels = [
        (loss, f"Training loss {size}")
        for loss, size in zip(training_losses_all, hidden_layer_sizes)
    ]
    lines_and_labels += [
        (loss, f"Validation loss {size}")
        for loss, size in zip(validation_losses_all, hidden_layer_sizes)
    ]
    plot_list_of_lines_and_labels(
        lines_and_labels=lines_and_labels,
        log=True,
        path=Path("experiments/experiment_15/loss.png"),
    )


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
        "--experiments", type=int, nargs="+", default=[], help="Experiments to run"
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
