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
import gpytorch
import argparse
import numpy as np
from pathlib import Path
import os

from src.dataset import (
    ParityTask,
    HiddenDataset,
    PeekParityTask,
    ModuloAdditionTask,
    ModuloSubtractionTask,
    ModuloDivisionTask,
    PolynomialTask,
    PolynomialTaskTwo,
    ModuloMultiplicationDoubleXTask,
    combine_datasets,
)
from src.model import TinyModel, ExpandableModel, ExactGPModel, ExactMarginalLikelihood
from src.train import train_model, Observer
from src.plot import (
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
    output_size = 2
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
        output_size=output_size,
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


def experiment_data_sensitivity(args):
    """
    Here we are going to look at the sensitivity of grokking to the size of the underlying dataset
    for the parity prediction task. We will use the same model as in experiment grokking_plain.
    """

    weight_decay = 1e-2
    learning_rate = 1e-1
    batch_size = 32
    input_size = 40
    output_size = 2
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
            output_size=output_size,
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
    output_size = 2
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
            output_size=output_size,
            random_seed=random_seed,
        )

        model.freeze([1])

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
    output_size = 2
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
        output_size=output_size,
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
    output_size = 2
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
        output_size=output_size,
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
    output_size = 2
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
        output_size=output_size,
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
    output_size = 2
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
            output_size=output_size,
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


def experiment_grokking_on_modulo_arithmetic(args):
    """
    Can we reproduce grokking within modulo arithmetic?
    """

    weight_decay = 1e-2
    learning_rate = 1e-1
    batch_size = 32
    p = 29
    input_size = p * 2
    output_size = p
    hidden_size = 1000
    epochs = 500
    number_training_samples = 1000
    number_validation_samples = 400
    random_seed = 0

    entire_dataset = ModuloAdditionTask(
        num_samples=number_training_samples + number_validation_samples,
        modulo=p,
        random_seed=random_seed,
    )

    training_dataset, validation_dataset = torch.utils.data.random_split(
        entire_dataset,
        [number_training_samples, number_validation_samples],
    )

    model = TinyModel(
        input_size=input_size,
        hidden_layer_size=hidden_size,
        output_size=output_size,
        random_seed=random_seed,
    )

    observer = Observer(
        observation_settings={"weights": {"frequency": 10, "layers": [1]}},
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

    observer.plot_me(path=Path("experiments/grokking_on_modulo_arithmetic/"), log=False)


def experiment_combined_prediction(args):
    """
    Combined prediction task of both parity and modulo addition.
    """

    weight_decay = 1e-2
    learning_rate = 1e-1
    batch_size = 32
    k = 3
    p = 2
    output_size = 2 + p
    input_size = 80
    hidden_size = 264
    epochs = 200
    number_training_samples = 2000
    number_validation_samples = 200
    random_seed = 0

    parity_dataset = ParityTask(
        sequence_length=k,
        num_samples=number_training_samples + number_validation_samples,
        random_seed=0,
    )

    hidden_parity_dataset = HiddenDataset(
        dataset=parity_dataset,
        total_length=int(input_size / 2),
        random_seed=0,
    )

    modulo_dataset = ModuloAdditionTask(
        num_samples=number_training_samples + number_validation_samples,
        modulo=p,
        random_seed=0,
    )

    hidden_modulo_dataset = HiddenDataset(
        dataset=modulo_dataset,
        total_length=int(input_size / 2),
        random_seed=0,
    )

    combined_dataset = combine_datasets(
        hidden_parity_dataset, hidden_modulo_dataset, individual=True
    )

    training_dataset, validation_dataset = torch.utils.data.random_split(
        combined_dataset,
        [2 * number_training_samples, 2 * number_validation_samples],
    )

    # Isolate the generalisation datasets
    indices_of_parity_dataset = [
        i
        for i, x in enumerate(validation_dataset)
        if (x[0][0 : int(input_size / 2)] == torch.zeros(int(input_size / 2))).all()
    ]

    print(f"Found {len(indices_of_parity_dataset)} samples in the parity dataset.")

    parity_prediction_subset = torch.utils.data.Subset(
        validation_dataset, indices_of_parity_dataset
    )

    indices_of_modulo_dataset = [
        i
        for i, x in enumerate(validation_dataset)
        if (x[0][int(input_size / 2) :] == torch.zeros(int(input_size / 2))).all()
    ]

    print(f"Found {len(indices_of_modulo_dataset)} samples in the modulo dataset.")

    modulo_prediction_subset = torch.utils.data.Subset(
        validation_dataset, indices_of_modulo_dataset
    )

    model = TinyModel(
        input_size=input_size,
        hidden_layer_size=hidden_size,
        output_size=output_size,
        random_seed=random_seed,
    )

    observer = Observer(
        observation_settings={"weights": {"frequency": 10, "layers": [1]}},
        generalisation_datasets={
            "parity": parity_prediction_subset,
            "modulo": modulo_prediction_subset,
        },
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

    observer.plot_me(path=Path("experiments/combined_prediction/"), log=False)


def experiment_combined_prediction_constrained(args):
    """
    Same experiment as the combined prediction task but a continual reduction
    in the hidden layer size.

    Currently, not getting grokking but some interesting weight sharing
    thing going on in 32.
    """

    weight_decay = 1e-2
    learning_rate = 1e-1
    batch_size = 32
    k = 3
    p = 2
    output_size = 2 + p
    input_size = 80
    hidden_size = 264
    epochs = 200
    number_training_samples = 2000
    number_validation_samples = 200
    random_seed = 0

    parity_dataset = ParityTask(
        sequence_length=k,
        num_samples=number_training_samples + number_validation_samples,
        random_seed=0,
    )

    hidden_parity_dataset = HiddenDataset(
        dataset=parity_dataset,
        total_length=int(input_size / 2),
        random_seed=0,
    )

    modulo_dataset = ModuloAdditionTask(
        num_samples=number_training_samples + number_validation_samples,
        modulo=p,
        random_seed=0,
    )

    hidden_modulo_dataset = HiddenDataset(
        dataset=modulo_dataset,
        total_length=int(input_size / 2),
        random_seed=0,
    )

    combined_dataset = combine_datasets(
        hidden_parity_dataset, hidden_modulo_dataset, individual=True
    )

    training_dataset, validation_dataset = torch.utils.data.random_split(
        combined_dataset,
        [2 * number_training_samples, 2 * number_validation_samples],
    )

    # Isolate the generalisation datasets
    indices_of_parity_dataset = [
        i
        for i, x in enumerate(validation_dataset)
        if (x[0][0 : int(input_size / 2)] == torch.zeros(int(input_size / 2))).all()
    ]

    print(f"Found {len(indices_of_parity_dataset)} samples in the parity dataset.")

    parity_prediction_subset = torch.utils.data.Subset(
        validation_dataset, indices_of_parity_dataset
    )

    indices_of_modulo_dataset = [
        i
        for i, x in enumerate(validation_dataset)
        if (x[0][int(input_size / 2) :] == torch.zeros(int(input_size / 2))).all()
    ]

    print(f"Found {len(indices_of_modulo_dataset)} samples in the modulo dataset.")

    modulo_prediction_subset = torch.utils.data.Subset(
        validation_dataset, indices_of_modulo_dataset
    )

    hidden_sizes = [264, 256, 128, 64, 32]

    for hidden_size in hidden_sizes:
        print(f"Hidden size: {hidden_size}")

        observer = Observer(
            observation_settings={"weights": {"frequency": 10, "layers": [1]}},
            generalisation_datasets={
                "parity": parity_prediction_subset,
                "modulo": modulo_prediction_subset,
            },
        )

        model = TinyModel(
            input_size=input_size,
            hidden_layer_size=hidden_size,
            output_size=output_size,
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
            observer=observer,
        )

        observer.plot_me(
            path=Path(f"experiments/combined_prediction_constrained/{hidden_size}"),
            log=False,
        )


def experiment_combined_hidden_and_data_constrained(args):
    """
    Same experiment as the combined prediction task but a continual reduction
    in the hidden layer size and a smaller number of training points.
    """

    weight_decay = 1e-2
    learning_rate = 1e-1
    batch_size = 32
    k = 3
    p = 2
    output_size = 2 + p
    input_size = 80
    hidden_size = 264
    epochs = 200
    number_training_samples = 700
    number_validation_samples = 200
    random_seed = 0

    parity_dataset = ParityTask(
        sequence_length=k,
        num_samples=number_training_samples + number_validation_samples,
        random_seed=0,
    )

    hidden_parity_dataset = HiddenDataset(
        dataset=parity_dataset,
        total_length=int(input_size / 2),
        random_seed=0,
    )

    modulo_dataset = ModuloAdditionTask(
        num_samples=number_training_samples + number_validation_samples,
        modulo=p,
        random_seed=0,
    )

    hidden_modulo_dataset = HiddenDataset(
        dataset=modulo_dataset,
        total_length=int(input_size / 2),
        random_seed=0,
    )

    combined_dataset = combine_datasets(
        hidden_parity_dataset, hidden_modulo_dataset, individual=True
    )

    training_dataset, validation_dataset = torch.utils.data.random_split(
        combined_dataset,
        [2 * number_training_samples, 2 * number_validation_samples],
    )

    # Isolate the generalisation datasets
    indices_of_parity_dataset = [
        i
        for i, x in enumerate(validation_dataset)
        if (x[0][0 : int(input_size / 2)] == torch.zeros(int(input_size / 2))).all()
    ]

    print(f"Found {len(indices_of_parity_dataset)} samples in the parity dataset.")

    parity_prediction_subset = torch.utils.data.Subset(
        validation_dataset, indices_of_parity_dataset
    )

    indices_of_modulo_dataset = [
        i
        for i, x in enumerate(validation_dataset)
        if (x[0][int(input_size / 2) :] == torch.zeros(int(input_size / 2))).all()
    ]

    print(f"Found {len(indices_of_modulo_dataset)} samples in the modulo dataset.")

    modulo_prediction_subset = torch.utils.data.Subset(
        validation_dataset, indices_of_modulo_dataset
    )

    hidden_sizes = [264, 256, 128, 64, 32]

    for hidden_size in hidden_sizes:
        print(f"Hidden size: {hidden_size}")

        observer = Observer(
            observation_settings={"weights": {"frequency": 10, "layers": [1]}},
            generalisation_datasets={
                "parity": parity_prediction_subset,
                "modulo": modulo_prediction_subset,
            },
        )

        model = TinyModel(
            input_size=input_size,
            hidden_layer_size=hidden_size,
            output_size=output_size,
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
            observer=observer,
        )

        observer.plot_me(
            path=Path(
                f"experiments/combined_hidden_and_data_constrained/{hidden_size}"
            ),
            log=False,
        )


def experiment_dependence_on_random_length(args):
    """
    Looking at dependence of grokking on the random feature length, presumably
    the longer the random features the more grokking.

    We complete this experiment with:
        * Modulo addition task
        * Parity prediction task
    """

    os.makedirs("experiments/dependence_on_random_length/parity/", exist_ok=True)
    os.makedirs("experiments/dependence_on_random_length/modulo/", exist_ok=True)

    weight_decay = 1e-2
    learning_rate = 1e-1
    batch_size = 32
    k = 6
    p = 3
    input_size = 6
    hidden_size = 264
    epochs = 100
    number_training_samples = 700
    number_validation_samples = 200
    random_seed = 0

    parity_dataset = ParityTask(
        sequence_length=k,
        num_samples=number_training_samples + number_validation_samples,
        random_seed=random_seed,
    )

    modulo_dataset = ModuloAdditionTask(
        num_samples=number_training_samples + number_validation_samples,
        modulo=p,
        random_seed=random_seed,
    )

    random_feature_length = [6, 10, 20, 30]

    parity_observers = []
    modulo_observers = []

    for hidden_length in random_feature_length:
        hidden_parity_dataset = HiddenDataset(
            dataset=parity_dataset, total_length=hidden_length, random_seed=random_seed
        )

        (
            hidden_parity_training,
            hidden_parity_validation,
        ) = torch.utils.data.random_split(
            hidden_parity_dataset,
            [number_training_samples, number_validation_samples],
        )

        hidden_modulo_dataset = HiddenDataset(
            dataset=modulo_dataset, total_length=hidden_length, random_seed=random_seed
        )

        (
            hidden_modulo_training,
            hidden_modulo_validation,
        ) = torch.utils.data.random_split(
            hidden_modulo_dataset,
            [number_training_samples, number_validation_samples],
        )

        parity_model = TinyModel(
            input_size=hidden_length,
            hidden_layer_size=hidden_size,
            output_size=2,
            random_seed=random_seed,
        )

        modulo_model = TinyModel(
            input_size=hidden_length,
            hidden_layer_size=hidden_size,
            output_size=p,
            random_seed=random_seed,
        )

        (parity_model, parity_observer) = train_model(
            training_dataset=hidden_parity_training,
            validation_dataset=hidden_parity_validation,
            model=parity_model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            loss_function_label="cross-entropy",
            optimiser_function_label="sgd",
            progress_bar=True,
        )

        parity_observers.append(parity_observer)

        (modulo_model, modulo_observer) = train_model(
            training_dataset=hidden_modulo_training,
            validation_dataset=hidden_modulo_validation,
            model=modulo_model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            loss_function_label="cross-entropy",
            optimiser_function_label="sgd",
            progress_bar=True,
        )

        modulo_observers.append(modulo_observer)

    plot_validation_and_accuracy_from_observers(
        parity_observers,
        random_feature_length,
        Path("experiments/dependence_on_random_length/parity/"),
    )
    plot_validation_and_accuracy_from_observers(
        modulo_observers,
        random_feature_length,
        Path("experiments/dependence_on_random_length/modulo/"),
    )


def experiment_dependence_on_weight_init(args):
    """
    We decrease the magntiude of initiasl weights to try and modify grokking.

    This experiment relates to the paper https://arxiv.org/abs/2210.01117,
    specifically figure 1b on page 2. If we haven't misunderstood, according
    to that figure if we decrease the weight norm to be very small we should not
    see grokking.
    """

    os.makedirs("experiments/dependence_on_weight_init/parity/", exist_ok=True)
    os.makedirs("experiments/dependence_on_weight_init/modulo/", exist_ok=True)

    weight_decay = 1e-2
    learning_rate = 1e-1
    batch_size = 32
    k = 3
    p = 3
    total_length = 40
    hidden_size = 264
    epochs = 200
    number_training_samples = 700
    number_validation_samples = 200
    random_seed = 0

    parity_dataset = ParityTask(
        sequence_length=k,
        num_samples=number_training_samples + number_validation_samples,
        random_seed=random_seed,
    )

    modulo_dataset = ModuloAdditionTask(
        num_samples=number_training_samples + number_validation_samples,
        modulo=p,
        random_seed=random_seed,
    )

    weight_norms = [1e-8, 1e-6, 1e-4, 1e-2, 1]

    parity_observers = []
    modulo_observers = []

    for weight_norm in weight_norms:
        hidden_parity_dataset = HiddenDataset(
            dataset=parity_dataset, total_length=total_length, random_seed=random_seed
        )

        (
            hidden_parity_training,
            hidden_parity_validation,
        ) = torch.utils.data.random_split(
            hidden_parity_dataset,
            [number_training_samples, number_validation_samples],
        )

        hidden_modulo_dataset = HiddenDataset(
            dataset=modulo_dataset, total_length=total_length, random_seed=random_seed
        )

        (
            hidden_modulo_training,
            hidden_modulo_validation,
        ) = torch.utils.data.random_split(
            hidden_modulo_dataset,
            [number_training_samples, number_validation_samples],
        )

        parity_model = TinyModel(
            input_size=total_length,
            hidden_layer_size=hidden_size,
            output_size=2,
            random_seed=random_seed,
        )

        # Multiply the weights by the weight norm size
        for param in parity_model.parameters():
            param.data *= weight_norm

        modulo_model = TinyModel(
            input_size=total_length,
            hidden_layer_size=hidden_size,
            output_size=p,
            random_seed=random_seed,
        )

        for param in modulo_model.parameters():
            param.data *= weight_norm

        # Print the weight norms
        print(
            f"Weight norm parity: {torch.norm(next(parity_model.parameters()))},\
                modulo: {torch.norm(next(modulo_model.parameters()))}"
        )

        (parity_model, parity_observer) = train_model(
            training_dataset=hidden_parity_training,
            validation_dataset=hidden_parity_validation,
            model=parity_model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            loss_function_label="cross-entropy",
            optimiser_function_label="sgd",
            progress_bar=True,
        )

        parity_observers.append(parity_observer)

        (modulo_model, modulo_observer) = train_model(
            training_dataset=hidden_modulo_training,
            validation_dataset=hidden_modulo_validation,
            model=modulo_model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            loss_function_label="cross-entropy",
            optimiser_function_label="sgd",
            progress_bar=True,
        )

        modulo_observers.append(modulo_observer)

    plot_validation_and_accuracy_from_observers(
        parity_observers,
        weight_norms,
        Path("experiments/dependence_on_weight_init/parity/"),
    )
    plot_validation_and_accuracy_from_observers(
        modulo_observers,
        weight_norms,
        Path("experiments/dependence_on_weight_init/modulo/"),
    )


def experiment_weight_magnitude_plot(args):
    """
    Here we want to plot the magntiude of weights inside of both the first
    and second weight matrix. Basically, does Grokking correspond to a
    large scale change in the weights?
    """

    weight_decay = 1e-2
    learning_rate = 1e-1
    batch_size = 32
    input_size = 40
    output_size = 2
    k = 3
    hidden_size = 200
    epochs = 300
    number_training_samples = 700
    number_validation_samples = 200

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

    model = TinyModel(
        input_size=input_size,
        hidden_layer_size=hidden_size,
        output_size=output_size,
        random_seed=random_seed,
    )

    observer = Observer(
        observation_settings={"weight_norm": {"frequency": 1, "layers": [1, 2]}},
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

    observer.plot_me(path=Path("experiments/weight_magnitude_plot/"), log=False)


def experiment_training_on_openai_datasets(args):
    """
    This experiment is to see if we can reproduce grokking or even get things
    to train on the openai datasets randomly choosen.

    These include: ModuloSubtractionTask, ModuloDivisionTask, PolynomialTask, PolynomialTaskTwo and the
    ModuloMultiplicationDoubleXTask.
    """

    os.makedirs("experiments/training_on_openai_datasets/", exist_ok=True)

    weight_decay = 1e-2
    learning_rate = 1e-1
    batch_size = 32
    input_size = 40
    p = 7
    hidden_size = 1000
    epochs = 200
    number_training_samples = 600
    number_validation_samples = 200
    random_seed = 0

    datasets = []

    for dataset in [
        ModuloSubtractionTask,
        ModuloDivisionTask,
        PolynomialTask,
        PolynomialTaskTwo,
        ModuloMultiplicationDoubleXTask,
    ]:
        datasets.append(
            dataset(
                num_samples=number_training_samples + number_validation_samples,
                random_seed=random_seed,
                modulo=p,
            )
        )

    observers_of_datasets = []

    for dataset in datasets:
        hidden_dataset = HiddenDataset(
            dataset=dataset,
            total_length=input_size,
            random_seed=random_seed,
        )

        training_dataset, validation_dataset = torch.utils.data.random_split(
            dataset,
            [number_training_samples, number_validation_samples],
        )

        (
            hidden_training_dataset,
            hidden_validation_dataset,
        ) = torch.utils.data.random_split(
            hidden_dataset,
            [number_training_samples, number_validation_samples],
        )

        model = TinyModel(
            input_size=2 * p,
            hidden_layer_size=hidden_size,
            output_size=p,
            random_seed=random_seed,
        )

        observer = Observer(
            observation_settings={"weight_norm": {"frequency": 1, "layers": [1, 2]}},
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

        observer.plot_me(
            path=Path(f"experiments/training_on_openai_datasets/{dataset.__name__()}/"),
        )

        # Hidden versions
        model_hidden = TinyModel(
            input_size=input_size,
            hidden_layer_size=hidden_size,
            output_size=p,
            random_seed=random_seed,
        )

        observer_hidden = Observer(
            observation_settings={"weight_norm": {"frequency": 1, "layers": [1, 2]}},
        )

        (model_hidden, observer_hidden) = train_model(
            training_dataset=hidden_training_dataset,
            validation_dataset=hidden_validation_dataset,
            model=model_hidden,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            loss_function_label="cross-entropy",
            optimiser_function_label="sgd",
            progress_bar=True,
            observer=observer_hidden,
        )

        observer_hidden.plot_me(
            path=Path(
                f"experiments/training_on_openai_datasets/{dataset.__name__()}_hidden/"
            ),
        )

        observers_of_datasets.append([observer, observer_hidden])

    for i, observer_list in enumerate(observers_of_datasets):
        os.makedirs(
            f"experiments/training_on_openai_datasets/{datasets[i].__name__()}/combined",
            exist_ok=True,
        )
        plot_validation_and_accuracy_from_observers(
            observer_list,
            ["regular", "hidden"],
            Path(
                f"experiments/training_on_openai_datasets/{datasets[i].__name__()}/combined"
            ),
        )


def experiment_grokking_plain_gp_regression(args):
    """
    Does grokking behaviour happens when using a diffirent model,
    GP regression?
    """

    import gpytorch
    from tqdm import tqdm
    import pdb
    from gpytorch.models import ApproximateGP
    from gpytorch.variational import CholeskyVariationalDistribution
    from gpytorch.variational import UnwhitenedVariationalStrategy
    from gpytorch.constraints import Positive
    import matplotlib.pyplot as plt

    torch.manual_seed(42)
    np.random.seed(42)

    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, x_train, y_train, likelihood):
            N, D = x_train.size(0), x_train.size(1)
            super(ExactGPModel, self).__init__(x_train, y_train, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    ard_num_dims=D,
                    lengthscale_constraint=Positive(torch.exp, torch.log),
                ),
                outputscale_constraint=Positive(torch.exp, torch.log),
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    x = torch.from_numpy(np.random.rand(100, 1) * 3 - 1.5)
    y = torch.sin(2 * x) + torch.randn([100, 1]) * 0.1
    y = y.squeeze()

    learning_rate = 1e-1
    epochs = 500

    x_train, y_train = x[:50, :], y[:50]
    x_valid, y_valid = x[50:, :], y[50:]

    plt.figure()
    plt.plot(x_train, y_train, "+k")
    plt.savefig("tmp/gpr_data.png")

    # initialize likelihood and model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(x_train, y_train, likelihood)
    model.to(device)
    likelihood.to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_valid = x_valid.to(device)
    y_valid = y_valid.to(device)

    model.covar_module.base_kernel.lengthscale = 0.001

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    train_mses, valid_mses = [], []
    train_lps, valid_lps = [], []
    lengthscales = []
    vfes = []
    epochs_ls = [0, 10, 50, 150]
    for i in tqdm(range(epochs)):
        optimizer.zero_grad()
        output = model(x_train)
        loss = -mll(output, y_train)
        loss.backward()

        model.eval()
        output = model(x_train)
        train_preds = likelihood(output)
        valid_output = model(x_valid)
        valid_preds = likelihood(valid_output)
        model.train()

        train_mse = ((output.mean - y_train) ** 2).mean().detach().cpu()
        valid_mse = ((valid_output.mean - y_valid) ** 2).mean().detach().cpu()
        train_lp = train_preds.log_prob(y_train).mean().detach().cpu()
        valid_lp = valid_preds.log_prob(y_valid).mean().detach().cpu()
        train_mses.append(train_mse)
        train_lps.append(train_lp)
        valid_mses.append(valid_mse)
        valid_lps.append(valid_lp)
        vfes.append(loss.detach().cpu())

        print(
            "Iter %d/%d - Loss: %.3f, train mse: %.3f, valid mse: %.3f, train lp: %.3f, valid lp %.3f"
            % (i + 1, epochs, loss.item(), train_mse, valid_mse, train_lp, valid_lp)
        )

        if i in epochs_ls:
            lengthscales.append(
                model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
            )
            model.eval()
            x = torch.from_numpy(np.linspace(-2, 2, 100)).to(device)
            plt.figure()
            plt.plot(x_train.cpu(), y_train.cpu(), "+k")
            pred = model(x)
            m, v = pred.mean.detach(), pred.variance.detach()
            plt.plot(x.cpu(), m.cpu(), "-b")
            plt.fill_between(
                x.cpu(),
                m.cpu() + 2 * torch.sqrt(v.cpu()),
                m.cpu() - 2 * torch.sqrt(v.cpu()),
                color="b",
                alpha=0.3,
            )

            plt.savefig("tmp/gpr_pred_%d.png" % i)

        optimizer.step()

    plt.figure()
    plt.plot(np.arange(epochs) + 1, train_mses, "-r", label="train")
    plt.plot(np.arange(epochs) + 1, valid_mses, "-b", label="validation")
    plt.xscale("log")
    plt.xlabel("epoch")
    plt.ylabel("mse")
    plt.legend()
    plt.savefig("tmp/gpr_mse.png")

    plt.figure()
    plt.plot(np.arange(epochs) + 1, train_lps, "-r", label="train")
    plt.plot(np.arange(epochs) + 1, valid_lps, "-b", label="validation")
    plt.xscale("log")
    plt.xlabel("epoch")
    plt.ylabel("log prob")
    plt.legend()
    plt.savefig("tmp/gpr_lp.png")

    plt.figure()
    plt.plot(np.arange(epochs) + 1, vfes, "-k")
    plt.xscale("log")
    plt.xlabel("epoch")
    plt.ylabel("NLML")
    plt.savefig("tmp/gpr_lml.png")


def experiment_grokking_gp_regression(args):
    """
    Does grokking behaviour happens when using a diffirent model, GP regression?
    """

    from tqdm import tqdm
    import pdb
    import matplotlib.pyplot as plt

    # TODO: deal with random seed
    torch.manual_seed(42)
    np.random.seed(42)

    # generate synthetic data
    true_noise_std = 0.1
    x = torch.from_numpy(np.random.rand(100, 1) * 3 - 1.5)
    y = torch.sin(2 * x) + torch.randn([100, 1]) * true_noise_std
    y = y.squeeze()
    x_train, y_train = x[:50, :], y[:50]
    x_valid, y_valid = x[50:, :], y[50:]

    # initialize likelihood and model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_valid = x_valid.to(device)
    y_valid = y_valid.to(device)

    no_steps = 200
    log_scale = torch.linspace(-5, 5, no_steps)
    log_lengthscale = torch.linspace(-7, 4, no_steps)
    ln, ll = torch.meshgrid(log_scale, log_lengthscale, indexing="xy")
    ml = torch.zeros_like(ln)
    fit_terms = torch.zeros_like(ln)
    complexity_terms = torch.zeros_like(ln)
    for i in tqdm(range(ln.shape[0])):
        for j in range(ln.shape[1]):
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise=true_noise_std**2
            )
            model = ExactGPModel(x_train, y_train, likelihood)
            mll = ExactMarginalLikelihood(likelihood, model)
            model.to(device)
            likelihood.to(device)
            model.covar_module.base_kernel.lengthscale = torch.exp(ll[i, j])
            model.covar_module.outputscale = torch.exp(ln[i, j])

            output = model(x_train)
            loss, fit, comp = mll(output, y_train)
            loss, fit, comp = -loss, -fit, -comp
            ml[i, j] = loss.detach().cpu()
            fit_terms[i, j] = fit.detach().cpu()
            complexity_terms[i, j] = comp.detach().cpu()

    plt.figure()
    plt.plot(x_train.cpu(), y_train.cpu(), "+k")
    plt.savefig("tmp/gpr_data.png")

    plt.figure(100)
    c = plt.pcolor(ln, ll, ml, cmap="RdBu")
    plt.colorbar(c)
    plt.xlabel("log outputscale")
    plt.ylabel("log lengthscale")
    plt.savefig("tmp/gpr_marginal_likelihood_landscape.png")

    plt.figure(101)
    c = plt.pcolor(ln, ll, fit_terms, cmap="RdBu")
    plt.colorbar(c)
    plt.xlabel("log outputscale")
    plt.ylabel("log lengthscale")

    plt.figure(102)
    c = plt.pcolor(ln, ll, complexity_terms, cmap="RdBu")
    plt.colorbar(c)
    plt.xlabel("log outputscale")
    plt.ylabel("log lengthscale")

    learning_rate = 1e-2
    epochs = 1000
    init_ll = [-6, -6, 3]
    init_ls = [-4, 4, 0]
    colors = ["g", "y", "b"]
    for init_idx, (ll_i, ls_i) in tqdm(enumerate(zip(init_ll, init_ls))):
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise=true_noise_std**2)
        model = ExactGPModel(x_train, y_train, likelihood)
        mll = ExactMarginalLikelihood(likelihood, model)
        model.to(device)
        likelihood.to(device)
        optimizer = torch.optim.Adam(
            model.covar_module.parameters(),
            lr=learning_rate,
        )
        model.covar_module.base_kernel.lengthscale = np.exp(ll_i)
        model.covar_module.outputscale = np.exp(ls_i)

        model.train()
        likelihood.train()

        train_mses, valid_mses = [], []
        train_lps, valid_lps = [], []
        lengthscales = []
        lmls, fits, comps = [], [], []
        epochs_ls = [0, 10, 50, 150, 250]
        ls_path = []
        ll_path = []

        for i in tqdm(range(epochs)):
            ls_path.append(torch.log(model.covar_module.outputscale).detach().cpu())
            ll_path.append(
                torch.log(model.covar_module.base_kernel.lengthscale[0, 0])
                .detach()
                .cpu()
            )
            optimizer.zero_grad()
            output = model(x_train)
            loss, fit, comp = mll(output, y_train)
            loss, fit, comp = -loss, -fit, -comp
            loss.backward()

            model.eval()
            output = model(x_train)
            train_preds = likelihood(output)
            valid_output = model(x_valid)
            valid_preds = likelihood(valid_output)
            model.train()

            train_mse = ((output.mean - y_train) ** 2).mean().detach().cpu()
            valid_mse = ((valid_output.mean - y_valid) ** 2).mean().detach().cpu()
            train_lp = train_preds.log_prob(y_train).mean().detach().cpu()
            valid_lp = valid_preds.log_prob(y_valid).mean().detach().cpu()
            train_mses.append(train_mse)
            train_lps.append(train_lp)
            valid_mses.append(valid_mse)
            valid_lps.append(valid_lp)
            lmls.append(loss.detach().cpu())
            fits.append(fit.detach().cpu())
            comps.append(comp.detach().cpu())

            print(
                "Iter %d/%d - Loss: %.3f, train mse: %.3f, valid mse: %.3f, train lp: %.3f, valid lp %.3f"
                % (i + 1, epochs, loss.item(), train_mse, valid_mse, train_lp, valid_lp)
            )

            if i in epochs_ls:
                lengthscales.append(
                    model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
                )
                model.eval()
                x = torch.from_numpy(np.linspace(-2, 2, 100)).to(device)
                plt.figure()
                plt.plot(x_train.cpu(), y_train.cpu(), "+k")
                pred = model(x)
                m, v = pred.mean.detach(), pred.variance.detach()
                plt.plot(x.cpu(), m.cpu(), "-b")
                plt.fill_between(
                    x.cpu(),
                    m.cpu() + 2 * torch.sqrt(v.cpu()),
                    m.cpu() - 2 * torch.sqrt(v.cpu()),
                    color="b",
                    alpha=0.3,
                )
                plt.ylim(-3, 3)

                plt.savefig("tmp/gpr_%d_pred_%d.png" % (init_idx, i))

            optimizer.step()

        plt.figure(100)
        plt.plot(ls_path, ll_path, color="k")
        plt.plot(ls_path[0], ll_path[0], "s", markersize=10, color=colors[init_idx])
        plt.plot(ls_path[-1], ll_path[-1], "*", markersize=10, color=colors[init_idx])
        plt.figure(101)
        plt.plot(ls_path, ll_path, color="k")
        plt.plot(ls_path[0], ll_path[0], "s", markersize=10, color=colors[init_idx])
        plt.plot(ls_path[-1], ll_path[-1], "*", markersize=10, color=colors[init_idx])
        plt.figure(102)
        plt.plot(ls_path, ll_path, color="k")
        plt.plot(ls_path[0], ll_path[0], "s", markersize=10, color=colors[init_idx])
        plt.plot(ls_path[-1], ll_path[-1], "*", markersize=10, color=colors[init_idx])

        plt.figure()
        plt.plot(np.arange(epochs) + 1, train_mses, "-r", label="train")
        plt.plot(np.arange(epochs) + 1, valid_mses, "-b", label="validation")
        plt.xscale("log")
        plt.xlabel("epoch")
        plt.ylabel("mse")
        plt.legend()
        plt.savefig("tmp/gpr_%d_mse.png" % init_idx)

        plt.figure()
        plt.plot(np.arange(epochs) + 1, train_lps, "-r", label="train")
        plt.plot(np.arange(epochs) + 1, valid_lps, "-b", label="validation")
        plt.xscale("log")
        plt.xlabel("epoch")
        plt.ylabel("log prob")
        plt.legend()
        plt.savefig("tmp/gpr_%d_lp.png" % init_idx)

        plt.figure()
        plt.plot(np.arange(epochs) + 1, lmls, "-k")
        plt.plot(np.arange(epochs) + 1, fits, "-b")
        plt.plot(np.arange(epochs) + 1, comps, "-r")
        plt.xscale("log")
        plt.xlabel("epoch")
        plt.ylabel("objective, data fit and complexity")
        plt.savefig("tmp/gpr_%d_lml.png" % init_idx)

    plt.figure(100)
    plt.savefig("tmp/gpr_marginal_likelihood_landscape.png")
    plt.figure(101)
    plt.savefig("tmp/gpr_datafit_landscape.png")
    plt.figure(102)
    plt.savefig("tmp/gpr_complexity_landscape.png")


def experiment_grokking_plain_gp_classification_toy(args):
    """
    Does grokking behaviour happens when using a diffirent model,
    GP classification?
    """

    torch.manual_seed(42)
    np.random.seed(42)
    import gpytorch
    from tqdm import tqdm
    import pdb
    from gpytorch.models import ApproximateGP
    from gpytorch.variational import CholeskyVariationalDistribution
    from gpytorch.variational import UnwhitenedVariationalStrategy
    import matplotlib.pyplot as plt

    class GPModel(ApproximateGP):
        def __init__(self, x_train):
            N, D = x_train.size(0), x_train.size(1)
            var_dist = CholeskyVariationalDistribution(N)
            var_stra = UnwhitenedVariationalStrategy(
                self, x_train, var_dist, learn_inducing_locations=False
            )
            super(GPModel, self).__init__(var_stra)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=D)
            )

        def forward(self, x):
            x_mean = self.mean_module(x)
            x_covar = self.covar_module(x)
            latent_pred = gpytorch.distributions.MultivariateNormal(x_mean, x_covar)
            return latent_pred

    x = torch.from_numpy(np.random.rand(100, 1) - 0.5).to(torch.float)
    y = ((x > 0) * 1).to(torch.float)
    y = y.squeeze()

    x_train, y_train = x[:20, :], y[:20]
    x_valid, y_valid = x[80:, :], y[80:]

    learning_rate = 5e-3
    epochs = 2000

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GPModel(x_train)
    likelihood = gpytorch.likelihoods.BernoulliLikelihood()

    model.covar_module.base_kernel.lengthscale = 0.005

    model.to(device)
    likelihood.to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_valid = x_valid.to(device)
    y_valid = y_valid.to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, y_train.numel())
    train_accs, valid_accs = [], []
    train_lps, valid_lps = [], []
    lengthscales = []
    vfes = []
    epochs_ls = [0, 10, 50, 500]
    # epochs_ls = [0, 1, 2]
    for i in tqdm(range(epochs)):
        optimizer.zero_grad()
        output = model(x_train)
        loss = -mll(output, y_train)
        loss.backward()

        model.eval()
        output = model(x_train)
        train_preds = likelihood(output)
        valid_output = model(x_valid)
        valid_preds = likelihood(valid_output)
        model.train()

        train_pred_class = train_preds.mean.ge(0.5)
        valid_pred_class = valid_preds.mean.ge(0.5)
        train_acc = (train_pred_class == y_train).sum() / x_train.shape[0]
        valid_acc = (valid_pred_class == y_valid).sum() / x_valid.shape[0]
        train_lp = train_preds.log_prob(y_train).mean()
        valid_lp = valid_preds.log_prob(y_valid).mean()
        train_accs.append(train_acc.detach().cpu())
        train_lps.append(train_lp.detach().cpu())
        valid_accs.append(valid_acc.detach().cpu())
        valid_lps.append(valid_lp.detach().cpu())
        vfes.append(loss.detach().cpu())

        print(
            "Iter %d/%d - Loss: %.3f, train acc: %.3f, valid acc: %.3f, train lp: %.3f, valid lp %.3f"
            % (i + 1, epochs, loss.item(), train_acc, valid_acc, train_lp, valid_lp)
        )

        if i in epochs_ls:
            lengthscales.append(
                model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
            )

            model.eval()
            x = torch.from_numpy(np.linspace(-1, 1, 300)).to(torch.float).to(device)
            plt.figure()
            plt.plot(x_train.cpu(), y_train.cpu(), "+k")
            pred = model(x)
            m, v = pred.mean.detach(), pred.variance.detach()
            plt.plot(x.cpu(), m.cpu(), "-b")
            plt.fill_between(
                x.cpu(),
                m.cpu() + 2 * torch.sqrt(v.cpu()),
                m.cpu() - 2 * torch.sqrt(v.cpu()),
                color="b",
                alpha=0.3,
            )
            observed_pred = likelihood(pred)
            pred_labels = observed_pred.mean.ge(0.5).detach()
            plt.plot(x.cpu(), pred_labels.cpu(), "-r")
            plt.ylim([-1, 2])

            plt.savefig("tmp/gpc_toy_pred_%d.png" % i)

        optimizer.step()

    plt.figure()
    plt.plot(np.arange(epochs) + 1, train_accs, "-r", label="train")
    plt.plot(np.arange(epochs) + 1, valid_accs, "-b", label="validation")
    for i, epoch in enumerate(epochs_ls):
        plt.axvline(epoch + 1, color="k", alpha=0.3)
    plt.xscale("log")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig("tmp/gpc_toy_acc.png")

    plt.figure()
    plt.plot(np.arange(epochs) + 1, train_lps, "-r", label="train")
    plt.plot(np.arange(epochs) + 1, valid_lps, "-b", label="validation")
    for i, epoch in enumerate(epochs_ls):
        plt.axvline(epoch + 1, color="k", alpha=0.3)
    plt.xscale("log")
    plt.xlabel("epoch")
    plt.ylabel("log prob")
    plt.legend()
    plt.savefig("tmp/gpc_toy_lp.png")

    plt.figure()
    plt.plot(np.arange(epochs) + 1, vfes, "-k")
    for i, epoch in enumerate(epochs_ls):
        plt.axvline(epoch + 1, color="k", alpha=0.3)
    plt.xscale("log")
    plt.xlabel("epoch")
    plt.ylabel("variational free energy")
    plt.savefig("tmp/gpc_toy_vfe.png")


def experiment_grokking_plain_gp_classification(args):
    """
    Does grokking behaviour happens when using a diffirent model,
    GP classification?
    """

    torch.manual_seed(42)
    np.random.seed(42)
    import gpytorch
    from tqdm import tqdm
    import pdb
    from gpytorch.models import ApproximateGP
    from gpytorch.variational import CholeskyVariationalDistribution
    from gpytorch.variational import UnwhitenedVariationalStrategy
    import matplotlib.pyplot as plt

    class GPModel(ApproximateGP):
        def __init__(self, x_train):
            N, D = x_train.size(0), x_train.size(1)
            var_dist = CholeskyVariationalDistribution(N)
            var_stra = UnwhitenedVariationalStrategy(
                self, x_train, var_dist, learn_inducing_locations=False
            )
            super(GPModel, self).__init__(var_stra)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=D)
            )

        def forward(self, x):
            x_mean = self.mean_module(x)
            x_covar = self.covar_module(x)
            latent_pred = gpytorch.distributions.MultivariateNormal(x_mean, x_covar)
            return latent_pred

    learning_rate = 1e-2
    input_size = 40
    k = 3
    epochs = 500
    number_training_samples = 1000
    number_validation_samples = 100
    random_seed = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    x_train = training_dataset.dataset.data[training_dataset.indices]
    y_train = training_dataset.dataset.targets[training_dataset.indices, 0]
    x_valid = validation_dataset.dataset.data[validation_dataset.indices]
    y_valid = validation_dataset.dataset.targets[validation_dataset.indices, 0]
    model = GPModel(x_train)
    likelihood = gpytorch.likelihoods.BernoulliLikelihood()

    # init_lengthscales = 5
    # current_lengthscale = model.covar_module.base_kernel.lengthscale
    # model.covar_module.base_kernel.lengthscale = init_lengthscales * torch.ones_like(current_lengthscale)

    model.to(device)
    likelihood.to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_valid = x_valid.to(device)
    y_valid = y_valid.to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, y_train.numel())
    train_accs, valid_accs = [], []
    train_lps, valid_lps = [], []
    lengthscales = []
    vfes = []
    epochs_ls = [10, 50, 150]
    # epochs_ls = [0, 1, 2]
    for i in tqdm(range(epochs)):
        optimizer.zero_grad()
        output = model(x_train)
        loss = -mll(output, y_train)
        loss.backward()

        train_preds = likelihood(output)
        valid_output = model(x_valid)
        valid_preds = likelihood(valid_output)
        train_pred_class = train_preds.mean.ge(0.5)
        valid_pred_class = valid_preds.mean.ge(0.5)
        train_acc = (
            train_pred_class == y_train
        ).sum().detach().cpu() / number_training_samples
        valid_acc = (
            valid_pred_class == y_valid
        ).sum().detach().cpu() / number_validation_samples
        train_lp = train_preds.log_prob(y_train).mean().detach().cpu()
        valid_lp = valid_preds.log_prob(y_valid).mean().detach().cpu()
        train_accs.append(train_acc)
        train_lps.append(train_lp)
        valid_accs.append(valid_acc)
        valid_lps.append(valid_lp)
        vfes.append(loss.detach().cpu())

        print(
            "Iter %d/%d - Loss: %.3f, train acc: %.3f, valid acc: %.3f, train lp: %.3f, valid lp %.3f"
            % (i + 1, epochs, loss.item(), train_acc, valid_acc, train_lp, valid_lp)
        )

        if i in epochs_ls:
            lengthscales.append(
                model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
            )

        optimizer.step()

    plt.figure()
    plt.plot(np.arange(epochs) + 1, train_accs, "-r", label="train")
    plt.plot(np.arange(epochs) + 1, valid_accs, "-b", label="validation")
    for i, epoch in enumerate(epochs_ls):
        plt.axvline(epoch, color="k", alpha=0.3)
    plt.xscale("log")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig("tmp/gpc_acc.png")

    plt.figure()
    plt.plot(np.arange(epochs) + 1, train_lps, "-r", label="train")
    plt.plot(np.arange(epochs) + 1, valid_lps, "-b", label="validation")
    for i, epoch in enumerate(epochs_ls):
        plt.axvline(epoch, color="k", alpha=0.3)
    plt.xscale("log")
    plt.xlabel("epoch")
    plt.ylabel("log prob")
    plt.legend()
    plt.savefig("tmp/gpc_lp.png")

    plt.figure()
    plt.plot(np.arange(epochs) + 1, vfes, "-k")
    for i, epoch in enumerate(epochs_ls):
        plt.axvline(epoch, color="k", alpha=0.3)
    plt.xscale("log")
    plt.xlabel("epoch")
    plt.ylabel("variational free energy")
    plt.savefig("tmp/gpc_vfe.png")

    fig, axes = plt.subplots(len(epochs_ls), 1, sharex=True)
    for i, epoch in enumerate(epochs_ls):
        axes[i].bar(
            np.arange(input_size), 1 / lengthscales[i][0, :], label="epoch %d" % epoch
        )
        axes[i].set_ylabel("1 / lengthscale")
        axes[i].legend()

    axes[-1].set_xlabel("input dim")
    plt.savefig("tmp/gpc_ls.png")


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

    for experiment_name in args.experiments:
        os.makedirs("experiments/experiment_name/", exist_ok=True)
        eval("experiment_{}(args)".format(experiment_name))
