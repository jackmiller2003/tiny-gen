"""
This file contains a list of experiments. The general paradigm to follow when writing
an experiment is to:
    1. Describe the experiment in the docstring
    2. Write the code to run the experiment
    3. Write the code to plot the results of the experiment

Somebody else using the code should be able to run the experiment by using the
relevant function in this file via the flag --experiment.

For reproducibility define a random seed or set of random seeds, passing them into:
    1. The dataset
    2. The model

You should be able to get exactly the same results as the author
by running the experiment.
"""

import argparse
import os
from pathlib import Path

import gpytorch
import numpy as np
import torch
import matplotlib.pyplot as plt
from gpytorch.constraints import Positive
from tqdm import tqdm

from src.dataset import (
    HiddenDataset,
    ModuloAdditionTask,
    ModuloDivisionTask,
    ModuloMultiplicationDoubleXTask,
    ModuloSubtractionTask,
    ParityTask,
    PeekParityTask,
    PolynomialTask,
    PolynomialTaskTwo,
    NoisySineWaveTask,
    combine_datasets,
    generate_zero_one_classification,
)
from src.model import (
    ExactGPModel,
    ExactMarginalLikelihood,
    ExpandableModel,
    TinyModel,
    ApproxGPModel,
)
from src.plot import plot_validation_and_accuracy_from_observers
from src.train import Observer, train_model, train_GP_model
from tools import plot_landsacpes_of_GP_model


def experiment_grokking_plain():
    """
    Can we recover grokking behaviour using cross-entropy loss
    from the hidden parity prediction paper: https://arxiv.org/abs/2303.11873.
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


def experiment_data_sensitivity():
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


def experiment_random_features():
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


def experiment_two_layer():
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


def experiment_random_feature_3_layer():
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


def experiment_double_grokking():
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


def experiment_rate_limiting():
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


def experiment_grokking_on_modulo_arithmetic():
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


def experiment_combined_prediction():
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


def experiment_combined_prediction_constrained():
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


def experiment_combined_hidden_and_data_constrained():
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


def experiment_dependence_on_random_length():
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


def experiment_dependence_on_weight_init():
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


def experiment_weight_magnitude_plot():
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


def experiment_training_on_openai_datasets():
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


def experiment_grokking_plain_gp_regression():
    """
    Does grokking behaviour happens when using a diffirent model,
    GP regression?
    """

    os.makedirs("experiments/grokking_plain_gp_regression", exist_ok=True)

    random_seed = 42
    learning_rate = 1e-1
    epochs = 300
    length_scale = 1e-3

    # Setting seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    dataset = NoisySineWaveTask(
        total_length=100,
        x_range=(-1.5, 1.5),
        amplitude=1,
        frequency=1 / (np.pi),
        phase=0,
        x_noise=0.1,
        y_noise=0.1,
        random_seed=random_seed,
        random_x=True,
    )

    training_dataset, validation_dataset = torch.utils.data.random_split(
        dataset,
        [50, 50],
    )

    train_inputs = torch.tensor(
        [x.clone().detach().unsqueeze(0) for x, y in training_dataset]
    )
    train_targets = torch.tensor(
        [y.clone().detach().unsqueeze(0) for x, y in training_dataset]
    )

    num_dimensions = 1

    plt.figure()
    plt.plot(train_inputs, train_targets, "+k")
    plt.savefig("tmp/gpr_data.png")

    # initialize likelihood and model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    train_inputs.to(device)
    train_targets.to(device)
    model = ExactGPModel(
        train_inputs=train_inputs,
        train_targets=train_targets,
        likelihood=likelihood,
        num_dimensions=num_dimensions,
    )
    model.covar_module.base_kernel.lengthscale = length_scale

    model.train()
    likelihood.train()

    model, observer = train_GP_model(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        model=model,
        learning_rate=learning_rate,
        epochs=epochs,
        loss_function_label="mse",
        optimiser_function_label="adam",
        likelihood=likelihood,
    )

    observer.plot_me(path=Path("experiments/grokking_plain_gp_regression/"), log=True)


def experiment_grokking_gp_regression_landscapes():
    """
    Can we examine the complexity and loss lanscape associated with grokking in the GP case?
    """

    # --- Setup ---#
    torch.manual_seed(42)
    np.random.seed(42)

    os.makedirs("experiments/grokking_gp_regression_landscapes", exist_ok=True)

    x_noise = 0.1
    random_seed = 42

    dataset = NoisySineWaveTask(
        total_length=100,
        x_range=(-1.5, 1.5),
        amplitude=1,
        frequency=1 / (np.pi),
        phase=0,
        x_noise=x_noise,
        y_noise=0.1,
        random_seed=random_seed,
        random_x=True,
    )

    # initialize likelihood and model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    training_dataset, validation_dataset = torch.utils.data.random_split(
        dataset,
        [50, 50],
    )

    train_inputs = torch.tensor(
        [x.clone().detach().unsqueeze(0) for x, y in training_dataset]
    ).to(device)

    train_targets = torch.tensor(
        [y.clone().detach().unsqueeze(0) for x, y in training_dataset]
    ).to(device)

    num_dimensions = 1

    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise=x_noise**2)
    model = ExactGPModel(
        train_inputs, train_targets, likelihood, num_dimensions=num_dimensions
    )

    plot_landsacpes_of_GP_model(
        training_dataset=training_dataset,
        model=model,
        likelihood=likelihood,
        path_to_plot=Path("experiments/grokking_gp_regression_landscapes/"),
        num_plotting_steps=100,
        epochs=3000,
        optimiser_function_label="adam",
        validation_dataset=validation_dataset,
        trajectories_through_landscape=True,
    )


def experiment_grokking_plain_gp_classification_toy():
    """
    Does grokking behaviour happens when using a diffirent model,
    GP classification?
    """

    path_of_experiment = Path("experiments/grokking_plain_gp_classification_toy")

    os.makedirs(path_of_experiment, exist_ok=True)

    verbose = False
    learning_rate = 5e-3
    epochs = 750

    all_train_accs, all_valid_accs, all_train_lps, all_valid_lps, all_vfes = (
        [],
        [],
        [],
        [],
        [],
    )

    for random_seed in tqdm([42, 52, 62, 72, 82]):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        x_train, y_train, x_valid, y_valid = generate_zero_one_classification(
            device, random_seed
        )

        model = ApproxGPModel(x_train)
        likelihood = gpytorch.likelihoods.BernoulliLikelihood()

        model.covar_module.base_kernel.lengthscale = 0.005

        model.to(device)
        likelihood.to(device)

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
        for i in range(epochs):
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

            if verbose:
                print(
                    "Iter %d/%d - Loss: %.3f, train acc: %.3f, valid acc: %.3f, train lp: %.3f, valid lp %.3f"
                    % (
                        i + 1,
                        epochs,
                        loss.item(),
                        train_acc,
                        valid_acc,
                        train_lp,
                        valid_lp,
                    )
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

        all_train_accs.append(train_accs)
        all_valid_accs.append(valid_accs)
        all_train_lps.append(train_lps)
        all_valid_lps.append(valid_lps)
        all_vfes.append(vfes)

    all_train_accs = np.array(all_train_accs)
    all_valid_accs = np.array(all_valid_accs)
    all_train_lps = np.array(all_train_lps)
    all_valid_lps = np.array(all_valid_lps)
    all_vfes = np.array(all_vfes)

    avg_train_accs, std_train_accs = all_train_accs.mean(axis=0), all_train_accs.std(
        axis=0
    )
    avg_valid_accs, std_valid_accs = all_valid_accs.mean(axis=0), all_valid_accs.std(
        axis=0
    )
    avg_train_lps, std_train_lps = all_train_lps.mean(axis=0), all_train_lps.std(axis=0)
    avg_valid_lps, std_valid_lps = all_valid_lps.mean(axis=0), all_valid_lps.std(axis=0)
    avg_vfes, std_vfes = all_vfes.mean(axis=0), all_vfes.std(axis=0)

    epoch_range = range(1, epochs + 1)
    plt.figure()
    plt.plot(epoch_range, avg_train_accs, "-r", label="train accuracy")
    plt.fill_between(
        epoch_range,
        avg_train_accs - std_train_accs,
        avg_train_accs + std_train_accs,
        color="r",
        alpha=0.3,
    )
    plt.plot(epoch_range, avg_valid_accs, "-b", label="validation accuracy")
    plt.fill_between(
        epoch_range,
        avg_valid_accs - std_valid_accs,
        avg_valid_accs + std_valid_accs,
        color="b",
        alpha=0.3,
    )

    plt.legend(loc="lower right")
    plt.xscale("log")
    plt.savefig(
        path_of_experiment / Path("gpc_toy_acc_averaged.pdf"), bbox_inches="tight"
    )

    plt.figure()

    # Plot the averaged VFE with the shaded area for standard deviation
    plt.plot(epoch_range, avg_vfes, "-k", label="Variational Free Energy (VFE)")
    plt.fill_between(
        epoch_range, avg_vfes - std_vfes, avg_vfes + std_vfes, color="k", alpha=0.3
    )

    # Configure plot
    plt.xlabel("Epoch")
    plt.ylabel("Variational Free Energy")
    plt.xscale("log")
    plt.legend(loc="lower right")

    # Save the plot
    plt.savefig(
        path_of_experiment / Path("gpc_toy_vfe_averaged.pdf"), bbox_inches="tight"
    )

    # Create a figure for Log Probabilities
    plt.figure()

    # Plot the averaged train log probabilities with the shaded area for standard deviation
    plt.plot(epoch_range, avg_train_lps, "-r", label="Train Log Probabilities")
    plt.fill_between(
        epoch_range,
        avg_train_lps - std_train_lps,
        avg_train_lps + std_train_lps,
        color="r",
        alpha=0.3,
    )

    # Plot the averaged validation log probabilities with the shaded area for standard deviation
    plt.plot(epoch_range, avg_valid_lps, "-b", label="Validation Log Probabilities")
    plt.fill_between(
        epoch_range,
        avg_valid_lps - std_valid_lps,
        avg_valid_lps + std_valid_lps,
        color="b",
        alpha=0.3,
    )

    # Configure plot
    plt.xlabel("Epoch")
    plt.ylabel("Log Probabilities")
    plt.xscale("log")
    plt.legend(loc="upper left")

    # Save the plot
    plt.savefig(
        path_of_experiment / Path("gpc_toy_log_probs_averaged.pdf"), bbox_inches="tight"
    )

    plt.figure()

    # Set up a subplot grid with 1 row and 2 columns, and double the width of the individual plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # First subplot for accuracies
    axes[0].plot(epoch_range, avg_train_accs, "-r", label="Train Accuracy")
    axes[0].fill_between(
        epoch_range,
        avg_train_accs - std_train_accs,
        avg_train_accs + std_train_accs,
        color="r",
        alpha=0.3,
    )
    axes[0].plot(epoch_range, avg_valid_accs, "-b", label="Validation Accuracy")
    axes[0].fill_between(
        epoch_range,
        avg_valid_accs - std_valid_accs,
        avg_valid_accs + std_valid_accs,
        color="b",
        alpha=0.3,
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="lower right")

    # Second subplot for log probabilities
    axes[1].plot(epoch_range, avg_train_lps, "-r", label="Train Log Probabilities")
    axes[1].fill_between(
        epoch_range,
        avg_train_lps - std_train_lps,
        avg_train_lps + std_train_lps,
        color="r",
        alpha=0.3,
    )
    axes[1].plot(epoch_range, avg_valid_lps, "-b", label="Validation Log Probabilities")
    axes[1].fill_between(
        epoch_range,
        avg_valid_lps - std_valid_lps,
        avg_valid_lps + std_valid_lps,
        color="b",
        alpha=0.3,
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Log Probabilities")
    axes[1].legend(loc="upper left")

    # Adjust space between subplots
    plt.tight_layout()

    # Save the combined plot
    plt.savefig(
        path_of_experiment / Path("gpc_toy_combined_averaged.pdf"), bbox_inches="tight"
    )

    # plt.figure()
    # plt.plot(np.arange(epochs) + 1, train_accs, "-r", label="train")
    # plt.plot(np.arange(epochs) + 1, valid_accs, "-b", label="validation")
    # for i, epoch in enumerate(epochs_ls):
    #     plt.axvline(epoch + 1, color="k", alpha=0.3)
    # plt.xscale("log")
    # plt.xlabel("epoch")
    # plt.ylabel("accuracy")
    # plt.legend()
    # plt.savefig("tmp/gpc_toy_acc.png")

    # plt.figure()
    # plt.plot(np.arange(epochs) + 1, train_lps, "-r", label="train")
    # plt.plot(np.arange(epochs) + 1, valid_lps, "-b", label="validation")
    # for i, epoch in enumerate(epochs_ls):
    #     plt.axvline(epoch + 1, color="k", alpha=0.3)
    # plt.xscale("log")
    # plt.xlabel("epoch")
    # plt.ylabel("log prob")
    # plt.legend()
    # plt.savefig("tmp/gpc_toy_lp.png")

    # plt.figure()
    # plt.plot(np.arange(epochs) + 1, vfes, "-k")
    # for i, epoch in enumerate(epochs_ls):
    #     plt.axvline(epoch + 1, color="k", alpha=0.3)
    # plt.xscale("log")
    # plt.xlabel("epoch")
    # plt.ylabel("variational free energy")
    # plt.savefig("tmp/gpc_toy_vfe.png")


def experiment_grokking_plain_gp_classification():
    """
    Does grokking behaviour happens when using a diffirent model,
    GP classification?
    """

    torch.manual_seed(42)
    np.random.seed(42)
    import pdb

    import gpytorch
    import matplotlib.pyplot as plt
    from gpytorch.models import ApproximateGP
    from gpytorch.variational import (
        CholeskyVariationalDistribution,
        UnwhitenedVariationalStrategy,
    )
    from tqdm import tqdm

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
        eval("experiment_{}()".format(experiment_name))
