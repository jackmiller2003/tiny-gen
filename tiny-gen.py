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
    RBFLinearModel,
    TinyLinearModel,
    TinyBayes,
)
from src.plot import plot_validation_and_accuracy_from_observers
from src.train import Observer, train_model, train_GP_model
from tools import (
    plot_landsacpes_of_GP_model,
    add_features_for_lr_classification,
    gaussian_loss,
    l2_norm_for_lr,
    accuracy_for_negative_positive,
    sample_in_subspace,
    plot_combined_relationship,
)
from src.common import get_accuracy
from sklearn.model_selection import train_test_split
import json
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import copy


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
        ParityTask,
        ModuloAdditionTask,
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


def experiment_grokking_via_concealment():
    """
    In this experiment we wish to show that concealment is an effective strategy for inducing grokking.
    """

    os.makedirs("experiments/grokking_via_concealment/", exist_ok=True)
    cache_file = "experiments/grokking_via_concealment/cache.json"

    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cache = json.load(f)
    else:
        cache = {}

    weight_decay = 1e-2
    learning_rate = 1e-1
    batch_size = 32
    p = 7
    hidden_size = 1000
    epochs = 300
    number_training_samples = 600
    number_validation_samples = 200

    threshold = 0.95

    additional_lengths = [0, 10, 20, 30, 40]

    datasets_to_test = [
        ModuloAdditionTask,
        ModuloSubtractionTask,
        # ModuloDivisionTask,
        # PolynomialTask,
        # PolynomialTaskTwo,
        # ModuloMultiplicationDoubleXTask,
    ]

    random_seeds = list(range(0, 3))

    array_of_gaps = np.zeros(
        (len(additional_lengths), len(random_seeds), len(datasets_to_test))
    )

    for i, additional_length in enumerate(additional_lengths):
        print(f"Working on additional length: {additional_length}")
        all_dataset_grokking_for_k = []

        for j, random_seed in enumerate(random_seeds):
            print(f"Working on random seed: {random_seed}")
            accuracies = []

            datasets = []

            for dataset in datasets_to_test:
                datasets.append(
                    dataset(
                        num_samples=number_training_samples + number_validation_samples,
                        random_seed=random_seed,
                        modulo=p,
                    )
                )

            for k, dataset in enumerate(datasets):
                found_key = False

                cache_name = f"{additional_length}_{random_seed}_{dataset.__name__()}"

                try:
                    cache[cache_name]
                    found_key = True
                except KeyError:
                    pass

                if found_key:
                    training_accuracy = cache[cache_name]["training_accuracy"]
                    validation_accuracy = cache[cache_name]["validation_accuracy"]

                else:
                    hidden_dataset = HiddenDataset(
                        dataset=dataset,
                        total_length=2 * p + additional_length,
                        random_seed=random_seed,
                    )

                    (
                        hidden_training_dataset,
                        hidden_validation_dataset,
                    ) = torch.utils.data.random_split(
                        hidden_dataset,
                        [number_training_samples, number_validation_samples],
                    )

                    model_hidden = TinyModel(
                        input_size=2 * p + additional_length,
                        hidden_layer_size=hidden_size,
                        output_size=p,
                        random_seed=random_seed,
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
                    )

                    training_accuracy = np.array(observer_hidden.training_accuracy)
                    validation_accuracy = np.array(observer_hidden.validation_accuracy)

                    cache[cache_name] = {
                        "training_accuracy": observer_hidden.training_accuracy,
                        "validation_accuracy": observer_hidden.validation_accuracy,
                    }

                training_indices = np.where(np.array(training_accuracy) > threshold)

                if training_indices[0].size > 0:
                    index_of_train = training_indices[0][0]
                else:
                    index_of_train = epochs

                validation_indices = np.where(np.array(validation_accuracy) > threshold)

                if validation_indices[0].size > 0:
                    index_of_val = validation_indices[0][0]
                else:
                    index_of_val = epochs

                print(f"Index of train: {index_of_train}")
                print(f"Index of val: {index_of_val}")

                array_of_gaps[i, j, k] = index_of_val - index_of_train

    with open(cache_file, "w") as f:
        json.dump(cache, f)

    print(f"Array of gaps: {array_of_gaps}")

    flattened_gaps = array_of_gaps.reshape(
        len(additional_lengths) * len(random_seeds) * len(datasets_to_test)
    )

    repeated_lengths = np.repeat(
        additional_lengths, len(random_seeds) * len(datasets_to_test)
    )

    coordinate_array = np.column_stack((repeated_lengths, flattened_gaps))

    print(f"Flattened gaps: {flattened_gaps}")
    print(f"Coordinate array: {coordinate_array}")

    ln_y = np.where(coordinate_array[:, 1] <= 0, 0, np.log(coordinate_array[:, 1]))

    # Perform linear fit
    coefficients = np.polyfit(coordinate_array[:, 0], ln_y, 1)

    # Extract exponential model parameters a and b
    a = coefficients[0]
    b = np.exp(coefficients[1])

    correlation_coefficient, p_value = pearsonr(coordinate_array[:, 0], ln_y)

    print(f"Total trend...")
    print(f"Correlation coefficient: {correlation_coefficient}")
    print(f"p-value: {p_value}")
    print(f"Exponential coefficients: a={a}, b={b}\n\n")

    # Plot the results
    plt.figure(figsize=(10, 6))

    dataset_names = [dataset.__name__ for dataset in datasets_to_test]

    x = np.linspace(
        min(coordinate_array[:, 0]), max(coordinate_array[:, 0]), 500
    )  # Generate smoother x values for curve
    y = b * np.exp(a * x)  # Calculate y-values for the exponential model
    plt.plot(
        x, y, "-", color="black", label="Exponential Fit"
    )  # 'r-' specifies a red line

    for i, dataset_name in enumerate(dataset_names):
        # Filter final_array to get rows related to the current dataset
        grokking_gaps_for_dataset = array_of_gaps[:, :, i]

        # Flatten like before

        grokking_gaps_for_dataset_flattened = grokking_gaps_for_dataset.reshape(
            len(additional_lengths) * len(random_seeds)
        )

        repeated_lengths = np.repeat(additional_lengths, len(random_seeds))

        coordinate_array_for_ds = np.column_stack(
            (repeated_lengths, grokking_gaps_for_dataset_flattened)
        )

        print(f"Grokking gap for dataset: {grokking_gaps_for_dataset}")

        # Calculate mean and std values from the grokking_gaps_for_dataset
        avg_values = np.mean(grokking_gaps_for_dataset, axis=1)
        std_values = np.std(grokking_gaps_for_dataset, axis=1)

        print(f"Average values: {avg_values}")
        print(f"Std values: {std_values}")

        plt.errorbar(
            additional_lengths,
            avg_values,
            yerr=std_values,
            fmt="o",
            label=dataset_name,
            capsize=5,
        )

        ln_y = np.where(
            coordinate_array_for_ds[:, 1] <= 0, 0, np.log(coordinate_array_for_ds[:, 1])
        )

        # Perform linear fit
        coefficients = np.polyfit(coordinate_array_for_ds[:, 0], ln_y, 1)

        # Extract exponential model parameters a and b
        a = coefficients[0]
        b = np.exp(coefficients[1])

        correlation_coefficient, p_value = pearsonr(coordinate_array_for_ds[:, 0], ln_y)

        print(f"Dataset: {dataset_name}")
        print(f"Correlation coefficient: {correlation_coefficient}")
        print(f"P-value: {p_value}\n\n")

    plt.xlabel("Additional Length")
    plt.ylabel("Grokking Gap")
    plt.legend(loc="best")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    plt.savefig(
        "experiments/grokking_via_concealment/grokking_vs_length.pdf",
        bbox_inches="tight",
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
    learning_rate = 1e-2
    epochs = 1500

    (
        all_train_accs,
        all_valid_accs,
        all_train_lps,
        all_valid_lps,
        all_complexities,
        all_data_fit,
        all_vfes,
    ) = (
        [],
        [],
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
        mll = gpytorch.mlls.VariationalELBO(
            likelihood, model, y_train.numel(), beta=1, combine_terms=True
        )
        mll_not_combined = gpytorch.mlls.VariationalELBO(
            likelihood, model, y_train.numel(), beta=1, combine_terms=False
        )
        train_accs, valid_accs = [], []
        train_lps, valid_lps = [], []
        lengthscales = []
        complexities = []
        data_fits = []
        vfes = []
        epochs_ls = [0, 10, 50, 500]
        # epochs_ls = [0, 1, 2]
        for i in range(epochs):
            optimizer.zero_grad()
            output = model(x_train)
            mll_out = mll_not_combined(output, y_train)
            data_fit = mll_out[0]
            complexity = mll_out[1]
            complexities.append(complexity.detach().cpu())
            data_fits.append(-data_fit.detach().cpu())
            # print(f"Output: {mll_out}")
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
        all_complexities.append(complexities)
        all_data_fit.append(data_fits)
        all_vfes.append(vfes)

    all_train_accs = np.array(all_train_accs)
    all_valid_accs = np.array(all_valid_accs)
    all_train_lps = np.array(all_train_lps)
    all_valid_lps = np.array(all_valid_lps)
    all_complexities = np.array(all_complexities)
    all_data_fit = np.array(all_data_fit)
    all_vfes = np.array(all_vfes)

    avg_train_accs, std_train_accs = all_train_accs.mean(axis=0), all_train_accs.std(
        axis=0
    )
    avg_valid_accs, std_valid_accs = all_valid_accs.mean(axis=0), all_valid_accs.std(
        axis=0
    )
    avg_train_lps, std_train_lps = all_train_lps.mean(axis=0), all_train_lps.std(axis=0)
    avg_valid_lps, std_valid_lps = all_valid_lps.mean(axis=0), all_valid_lps.std(axis=0)
    avg_complexities, std_complexities = (
        all_complexities.mean(axis=0),
        all_complexities.std(axis=0),
    )
    avg_data_fit, std_data_fit = all_data_fit.mean(axis=0), all_data_fit.std(axis=0)
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
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

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

    # Second subplot for complexity
    axes[1].plot(epoch_range, avg_complexities, "-r", label="Train Complexity")
    axes[1].fill_between(
        epoch_range,
        avg_complexities - std_complexities,
        avg_complexities + std_complexities,
        color="r",
        alpha=0.3,
    )

    axes[1].plot(epoch_range, avg_data_fit, "-b", label="Negative Train Data Fit")
    axes[1].fill_between(
        epoch_range,
        avg_data_fit - std_data_fit,
        avg_data_fit + std_data_fit,
        color="b",
        alpha=0.3,
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Objective")
    axes[1].legend(loc="upper right")

    # Adjust space between subplots
    plt.tight_layout()

    # Save the combined plot
    plt.savefig(path_of_experiment / Path("gp_zero_one.pdf"), bbox_inches="tight")


def experiment_grokking_plain_gp_classification():
    """
    Does grokking behaviour happens when using a diffirent model,
    GP classification?
    """

    os.makedirs("experiments/grokking_plain_gp_classification", exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

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
    model = ApproxGPModel(x_train)
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

    fig, axes = plt.subplots(len(epochs_ls), 1, sharex=True, figsize=(14, 10))
    for i, epoch in enumerate(epochs_ls):
        axes[i].bar(
            np.arange(input_size), 1 / lengthscales[i][0, :], label="epoch %d" % epoch
        )
        axes[i].set_ylabel("Inverse lengthscale")
        axes[i].set_title("Epoch %d" % epoch)
        # axes[i].legend()

    axes[-1].set_xlabel("Input dimension")
    plt.savefig("tmp/gpc_ls.pdf", bbox_inches="tight")


def experiment_parity_gp_classification_batch():
    """
    Does grokking behaviour happens when using a diffirent model,
    GP classification?
    """

    path_of_experiment = Path("experiments/parity_gp_classification_batch")

    os.makedirs(path_of_experiment, exist_ok=True)

    learning_rate = 1e-2
    input_size = 40
    k = 3
    epochs = 500
    number_training_samples = 1000
    number_validation_samples = 100
    random_seed = 0

    verbose = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    (
        all_train_accs,
        all_valid_accs,
        all_train_lps,
        all_valid_lps,
        all_complexities,
        all_data_fit,
        all_vfes,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for random_seed in tqdm([42, 52, 62, 72, 82]):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

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

        x_train = training_dataset.dataset.data[training_dataset.indices].to(device)
        y_train = training_dataset.dataset.targets[training_dataset.indices, 0].to(
            device
        )
        x_valid = validation_dataset.dataset.data[validation_dataset.indices].to(device)
        y_valid = validation_dataset.dataset.targets[validation_dataset.indices, 0].to(
            device
        )
        model = ApproxGPModel(x_train)
        likelihood = gpytorch.likelihoods.BernoulliLikelihood()

        # init_lengthscales = 5
        # current_lengthscale = model.covar_module.base_kernel.lengthscale
        # model.covar_module.base_kernel.lengthscale = init_lengthscales * torch.ones_like(current_lengthscale)

        model.to(device)
        likelihood.to(device)

        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, y_train.numel())
        mll_not_combined = gpytorch.mlls.VariationalELBO(
            likelihood, model, y_train.numel(), combine_terms=False
        )
        train_accs, valid_accs = [], []
        train_lps, valid_lps = [], []
        lengthscales = []
        complexities = []
        data_fits = []
        vfes = []
        epochs_ls = [10, 50, 150]
        # epochs_ls = [0, 1, 2]
        for i in range(epochs):
            optimizer.zero_grad()
            output = model(x_train)
            mll_out = mll_not_combined(output, y_train)
            data_fit = mll_out[0]
            complexity = mll_out[1]
            complexities.append(complexity.detach().cpu())
            data_fits.append(-data_fit.detach().cpu())
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

            optimizer.step()

        all_train_accs.append(train_accs)
        all_valid_accs.append(valid_accs)
        all_train_lps.append(train_lps)
        all_valid_lps.append(valid_lps)
        all_complexities.append(complexities)
        all_data_fit.append(data_fits)
        all_vfes.append(vfes)

    all_train_accs = np.array(all_train_accs)
    all_valid_accs = np.array(all_valid_accs)
    all_train_lps = np.array(all_train_lps)
    all_valid_lps = np.array(all_valid_lps)
    all_complexities = np.array(all_complexities)
    all_data_fit = np.array(all_data_fit)
    all_vfes = np.array(all_vfes)

    avg_train_accs, std_train_accs = all_train_accs.mean(axis=0), all_train_accs.std(
        axis=0
    )
    avg_valid_accs, std_valid_accs = all_valid_accs.mean(axis=0), all_valid_accs.std(
        axis=0
    )
    avg_train_lps, std_train_lps = all_train_lps.mean(axis=0), all_train_lps.std(axis=0)
    avg_valid_lps, std_valid_lps = all_valid_lps.mean(axis=0), all_valid_lps.std(axis=0)
    avg_complexities, std_complexities = (
        all_complexities.mean(axis=0),
        all_complexities.std(axis=0),
    )
    avg_data_fit, std_data_fit = all_data_fit.mean(axis=0), all_data_fit.std(axis=0)
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
        path_of_experiment / Path("gp_parity_acc_averaged.pdf"), bbox_inches="tight"
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
        path_of_experiment / Path("gp_parity_vfe_averaged.pdf"), bbox_inches="tight"
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
        path_of_experiment / Path("gp_parity_log_probs_averaged.pdf"),
        bbox_inches="tight",
    )

    plt.figure()

    # Set up a subplot grid with 1 row and 2 columns, and double the width of the individual plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

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
    axes[1].plot(epoch_range, avg_complexities, "-r", label="Complexities")
    axes[1].fill_between(
        epoch_range,
        avg_complexities - std_complexities,
        avg_complexities + std_complexities,
        color="r",
        alpha=0.3,
    )
    axes[1].plot(epoch_range, avg_data_fit, "-b", label="Negative Data Fit")
    axes[1].fill_between(
        epoch_range,
        avg_data_fit - std_data_fit,
        avg_data_fit + std_data_fit,
        color="b",
        alpha=0.3,
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Log Probabilities")
    axes[1].legend(loc="upper right")

    # Adjust space between subplots
    plt.tight_layout()

    # Save the combined plot
    plt.savefig(
        path_of_experiment / Path("gp_parity.pdf"),
        bbox_inches="tight",
    )


def experiment_grokking_with_rbf_01_lr():
    """
    In this experiment, we hope to demonstrate grokking with the use of RBF kernels
    and bayesian linear regression.

    Some help was provided by: https://nbviewer.org/github/krasserm/bayesian-machine-learning/blob/dev/bayesian-linear-regression/bayesian_linear_regression.ipynb
    """

    # Dataset is same as toy GP case
    np.random.seed(42)
    torch.manual_seed(42)
    x = torch.from_numpy(np.random.rand(100, 1) - 0.5).to(torch.float)
    y = ((x > 0) * 1).to(torch.float)
    y = y.squeeze()
    n_train = 6

    # x_train = torch.from_numpy(
    #     np.array([-0.5, -0.4, -0.05, 0.05, 0.3, 0.5], dtype=np.float32)
    # )
    # x_train = x_train.unsqueeze(1)
    # y_train = torch.from_numpy(np.array([0, 0, 0, 1.0, 1.0, 1.0], dtype=np.float32))

    x_train, y_train = x[:n_train, :], y[:n_train]
    x_valid, y_valid = x[n_train:, :], y[n_train:]

    # model = RBFLinearModel(rbf_means=torch.linspace(-1, 1, 5000), rbf_variance=4e-4)
    model = RBFLinearModel(rbf_means=torch.linspace(-1, 1, 100), rbf_variance=0.0001)
    wd = 0.01
    # wd = 0.1
    optimiser = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=wd)

    train_loss = []
    val_loss = []
    train_accs = []
    val_accs = []
    l2_terms = []
    total_loss = []

    for epoch in tqdm(range(5000)):
        optimiser.zero_grad()

        outputs = model(x_train)
        # loss = torch.nn.functional.mse_loss(outputs.squeeze(), y_train)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs.squeeze(), y_train
        )
        l2_term = 0
        for param in model.parameters():
            l2_term += (param**2).sum() * wd
        l2_terms.append(l2_term.detach().numpy())
        # for param in model.parameters():
        #     loss += param.abs().sum() * 0.01
        acc = (((outputs.squeeze() > 0.5) == y_train) * 1.0).sum() / y_train.shape[0]

        train_loss.append(loss.detach().numpy())
        total_loss.append((loss + l2_term).detach().numpy())
        train_accs.append(acc.item())

        loss.backward()
        optimiser.step()

        model.eval()

        val_outputs = model(x_valid)
        # v_loss = torch.nn.functional.mse_loss(val_outputs.squeeze(), y_valid)
        v_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            val_outputs.squeeze(), y_valid
        )

        val_loss.append(v_loss.detach().numpy())
        val_acc = (
            ((val_outputs.squeeze() > 0.5) == y_valid) * 1.0
        ).sum() / y_valid.shape[0]
        val_accs.append(val_acc.item())
        print(loss.item(), v_loss.item())
        print(acc.item(), val_acc.item())

    plt.figure()

    plt.plot(train_loss, label="train")
    plt.plot(l2_terms, label="l2 term")
    plt.plot(total_loss, label="total")
    plt.plot(val_loss, label="validation")

    plt.yscale("log")
    plt.xscale("log")

    plt.legend()

    plt.savefig("tmp/grokking_with_rbf_lr_loss.png")

    plt.figure()

    plt.plot(train_accs, label="train")
    plt.plot(val_accs, label="validation")

    plt.yscale("log")
    plt.xscale("log")

    plt.legend()

    plt.savefig("tmp/grokking_with_rbf_lr_acc.png")

    # Plotting the inference
    model.eval()  # Make sure the model is in evaluation mode

    # Generate predictions over a range
    x_range = torch.linspace(-0.6, 0.6, 200).view(
        -1, 1
    )  # Extend a bit for visualization purposes
    # y_pred = model(x_range).detach().numpy()
    y_pred = torch.nn.functional.sigmoid(model(x_range)).detach().numpy()

    plt.figure()
    plt.scatter(
        x_valid.numpy(), y_valid.numpy(), color="red", s=50, label="Validation Data"
    )
    plt.scatter(
        x_train.numpy(), y_train.numpy(), color="blue", s=50, label="Train Data"
    )
    plt.plot(x_range.numpy(), y_pred.T, color="green", label="Model Inference")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Inference of the RBF Linear Model")
    plt.legend()

    plt.savefig("tmp/grokking_with_rbf_lr_inference.png")


def experiment_grokking_with_rbf_sin_lr():
    """
    In this experiment, we hope to demonstrate grokking with the use of RBF kernels on a sin
    curve (with noise).

    Some help was provided by: https://nbviewer.org/github/krasserm/bayesian-machine-learning/blob/dev/bayesian-linear-regression/bayesian_linear_regression.ipynb
    """

    # Dataset is same as toy GP case

    x = torch.from_numpy(np.sort(5 * np.random.rand(400, 1), axis=0))

    x = x.to(dtype=torch.float32)

    y = torch.sin(x).ravel()
    # y[::4] += 2 * (0.5 - np.random.rand(100))  # add noise

    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=0.4, random_state=42
    )

    y_train[::4] += 2 * (0.5 - np.random.rand(60))  # add noise

    model = RBFLinearModel(rbf_means=torch.linspace(0, 5, 100), rbf_variance=0.1)

    optimiser = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-2)

    train_loss = []
    val_loss = []

    for epoch in tqdm(range(500)):
        optimiser.zero_grad()

        outputs = model(x_train)

        loss = torch.nn.functional.mse_loss(outputs.squeeze(), y_train)

        train_loss.append(loss.detach().numpy())

        loss.backward()
        optimiser.step()

        model.eval()

        val_outputs = model(x_valid)
        v_loss = torch.nn.functional.mse_loss(val_outputs.squeeze(), y_valid)

        val_loss.append(v_loss.detach().numpy())

    plt.figure()

    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="validation")

    plt.yscale("log")

    plt.legend()

    plt.savefig("tmp/grokking_with_sin_rbf_lr_loss.png")

    # Plotting the inference
    model.eval()  # Make sure the model is in evaluation mode

    # Generate predictions over a range
    x_range = torch.linspace(0, 5, 500).view(
        -1, 1
    )  # Extend a bit for visualization purposes
    y_pred = model(x_range).detach().numpy()

    plt.figure()
    plt.scatter(
        x_train.numpy(), y_train.numpy(), color="blue", s=50, label="Train Data"
    )
    plt.scatter(
        x_valid.numpy(), y_valid.numpy(), color="red", s=50, label="Validation Data"
    )
    plt.plot(x_range.numpy(), y_pred.T, color="green", label="Model Inference")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Inference of the RBF Linear Model")
    plt.legend()

    plt.savefig("tmp/grokking_with_sin_rbf_lr_inference.png")


def experiment_grokking_lr_classification():
    """
    Can we induce grokking on linear regression?
    """

    path_of_experiment = Path("experiments/grokking_lr_classification")

    os.makedirs(path_of_experiment, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 5000

    all_train_accs, all_valid_accs, all_train_complexity = ([], [], [])

    for random_seed in tqdm(range(0, 5)):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        # we generate some data
        x = torch.from_numpy(np.random.rand(100, 1) - 0.5).to(torch.float)
        y = 0.3 * x
        y = y.to(torch.float)
        y = y.squeeze()
        n_train = 2

        # torch.manual_seed(random_seed)
        # np.random.seed(random_seed)

        x = add_features_for_lr_classification(x)

        # split for train / validation
        x_train, y_train = x[:n_train, :].to(device), y[:n_train].to(device)
        x_valid, y_valid = x[n_train:, :].to(device), y[n_train:].to(device)
        x_train_plot = x[:n_train, :].to(device)
        x_valid_plot = x[n_train:].to(device)

        # Extend a bit for visualization purposes
        x_plot = torch.linspace(-1, 1, 500).view(-1, 1)
        x_plot = add_features_for_lr_classification(x_plot)

        x_plot = x_plot.to(device)

        # we create a linear regression model
        input_size = x_train.shape[1]
        output_size = 1

        model = TinyLinearModel(input_size, output_size, random_seed)

        # change the init to induce grokking, we want to start in high complexity region
        model.fc1.weight.data = torch.tensor([[0.0005, 0.9, 0.9, 0.9]]).to(device)

        # move model to device
        model = model.to(device)

        optimiser = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=0)
        train_acc = []
        val_acc = []
        training_fit = []
        valid_fit = []
        training_total_loss = []
        training_reg = []

        plot_epochs = [0, 10, 100, 1000, 10000]
        for epoch in range(epochs):
            optimiser.zero_grad()
            output = model(x_train)
            data_fit = gaussian_loss(output.squeeze(), y_train) / n_train
            l2_term = l2_norm_for_lr(model, device) / n_train
            total_loss = data_fit + l2_term
            total_loss.backward()
            optimiser.step()

            training_reg.append(l2_term.item())
            training_fit.append(data_fit.item())
            training_total_loss.append(total_loss.item())

            val_outputs = model(x_valid)
            valid_data_fit = (
                gaussian_loss(val_outputs.squeeze(), y_valid) / x_valid.shape[0]
            )
            valid_fit.append(valid_data_fit.item())

            train_acc.append(accuracy_for_negative_positive(output.squeeze(), y_train))
            val_acc.append(
                accuracy_for_negative_positive(val_outputs.squeeze(), y_valid)
            )

            # if epoch in plot_epochs:
            #     plot_lr_pred(epoch, model, x_plot, x_valid, y_valid, x_train, y_train)

        all_train_accs.append(train_acc)
        all_valid_accs.append(val_acc)
        all_train_complexity.append(training_reg)

    all_train_accs = np.array(all_train_accs)
    all_valid_accs = np.array(all_valid_accs)
    all_train_complexity = np.array(all_train_complexity)

    avg_train_accs, std_train_accs = all_train_accs.mean(axis=0), all_train_accs.std(
        axis=0
    )

    avg_valid_accs, std_valid_accs = all_valid_accs.mean(axis=0), all_valid_accs.std(
        axis=0
    )

    avg_train_complexity, std_train_complexity = (
        all_train_complexity.mean(axis=0),
        all_train_complexity.std(axis=0),
    )

    epoch_range = range(1, epochs + 1)
    # Set up a subplot grid with 1 row and 2 columns, and double the width of the individual plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

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
    axes[1].plot(epoch_range, avg_train_complexity, "-r", label="Train Complexity")
    axes[1].fill_between(
        epoch_range,
        avg_train_complexity - std_train_complexity,
        avg_train_complexity + std_train_complexity,
        color="r",
        alpha=0.3,
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Train Complexity")
    axes[1].legend(loc="upper right")

    # Adjust space between subplots
    plt.tight_layout()

    # Save the combined plot
    plt.savefig(
        path_of_experiment / Path("grokking_classification_combined.pdf"),
        bbox_inches="tight",
    )


def experiment_training_with_vafe():
    """
    We look at the variational free energy and its components during grokking.
    """

    os.makedirs("experiments/training_with_vafe/", exist_ok=True)

    weight_decay = 0
    learning_rate = 1e-1
    batch_size = 32
    input_size = 3
    output_size = 2
    k = 3
    hidden_size = 200
    epochs = 500
    number_training_samples = 1000
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

    model = TinyBayes(
        input_size=input_size,
        hidden_layer_size=hidden_size,
        output_size=output_size,
        random_seed=random_seed,
        normalise_output=False,
    )

    observer = Observer(
        observation_settings={
            "variational_free_energy": {},
            "weights": {"frequency": 10, "layers": [1]},
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

    observer.plot_me(path=Path("experiments/training_with_vafe/"), log=True)


def experiment_grokking_plain_with_vafe():
    """
    We look at the variational free energy and its components during grokking.
    """

    os.makedirs("experiments/grokking_plain_with_vafe/", exist_ok=True)

    weight_decay = 0
    learning_rate = 1e-1
    batch_size = 32
    input_size = 30
    output_size = 2
    k = 3
    hidden_size = 200
    epochs = 500
    number_training_samples = 600
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

    model = TinyBayes(
        input_size=input_size,
        hidden_layer_size=hidden_size,
        output_size=output_size,
        random_seed=random_seed,
        normalise_output=False,
    )

    observer = Observer(
        observation_settings={
            "variational_free_energy": {},
        },
    )

    (model, observer) = train_model(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        model=model,
        learning_rate=learning_rate,
        weight_decay=0,
        epochs=epochs,
        batch_size=batch_size,
        loss_function_label="cross-entropy",
        optimiser_function_label="sgd",
        progress_bar=True,
        observer=observer,
    )

    observer.plot_me(path=Path("experiments/grokking_plain_with_vafe/"), log=False)


def experiment_grokking_accessibility_vafe():
    """
    We look at various random seeds and thus trajectories through the weight space of
    the network.
    """

    os.makedirs("experiments/grokking_accessibility_vafe/", exist_ok=True)

    weight_decay = 0
    learning_rate = 1e-1
    batch_size = 32
    input_size = 30
    output_size = 2
    k = 3
    hidden_size = 200
    epochs = 500
    number_training_samples = 600
    number_validation_samples = 200
    n_random_seeds = 1
    every_n_to_check_weights = 10

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

    all_vfes = []
    all_complexities = []
    all_datafit = []
    all_training_accuracy = []
    all_validation_accuracy = []

    for random_seed in range(0, n_random_seeds):
        model = TinyBayes(
            input_size=input_size,
            hidden_layer_size=hidden_size,
            output_size=output_size,
            random_seed=random_seed,
            normalise_output=False,
        )

        observer = Observer(
            observation_settings={
                "variational_free_energy": {},
                "weights": {"frequency": 10, "layers": [1, 2]},
            },
        )

        (model, observer) = train_model(
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            model=model,
            learning_rate=learning_rate,
            weight_decay=0,
            epochs=epochs,
            batch_size=batch_size,
            loss_function_label="cross-entropy",
            optimiser_function_label="sgd",
            progress_bar=True,
            observer=observer,
        )

        os.makedirs(
            f"experiments/grokking_accessibility_vafe/{random_seed}", exist_ok=True
        )

        all_vfes.append(observer.variational_free_energy)
        all_complexities.append(observer.complexity_loss)
        all_datafit.append(observer.error_loss)
        all_training_accuracy.append(observer.training_accuracy)
        all_validation_accuracy.append(observer.validation_accuracy)

        layer_1_weights = np.array(observer.weights[1])
        layer_2_weights = np.array(observer.weights[2])

        # Flatten the weights
        layer_1_weights = layer_1_weights.reshape(layer_1_weights.shape[0], -1)
        layer_2_weights = layer_2_weights.reshape(layer_2_weights.shape[0], -1)

        print(layer_1_weights.shape)
        print(layer_2_weights.shape)

        weight_vector = np.zeros(
            (
                layer_2_weights.shape[0],
                layer_1_weights.shape[1] + layer_2_weights.shape[1],
            )
        )

        weight_vector[:, : layer_1_weights.shape[1]] = layer_1_weights
        weight_vector[:, layer_1_weights.shape[1] :] = layer_2_weights

        if random_seed == 0:
            all_weights = np.zeros((n_random_seeds * epochs, weight_vector.shape[1]))

        print(weight_vector.shape)
        print(all_weights.shape)

        all_weights[
            random_seed * epochs : (random_seed + 1) * epochs, :
        ] = weight_vector

        observer.plot_me(
            path=Path(f"experiments/grokking_accessibility_vafe/{random_seed}"),
            log=False,
        )

    print(all_weights.shape)
    print("Completing PCA")

    # Standardize the data
    scaler = StandardScaler()
    all_weights_standardized = scaler.fit_transform(all_weights)

    # Apply PCA to get top 20 directions
    pca = PCA(n_components=20)
    principalComponents = pca.fit_transform(all_weights_standardized)

    top_20_directions = pca.components_

    # Convert numpy array to a list for JSON serialization
    top_20_directions_list = top_20_directions.tolist()

    # Save to a JSON file
    with open(
        "experiments/grokking_accessibility_vafe/top_20_pca_directions.json", "w"
    ) as json_file:
        json.dump(top_20_directions_list, json_file)

    # Save model to a pickle file
    with open(
        "experiments/grokking_accessibility_vafe/final_model.pkl", "wb"
    ) as pickle_file:
        pickle.dump(model, pickle_file)

    print(f"Finished PCA, saved to top_20_pca_directions.json")

    # Complete PCA on all_weights and save top 20 directions in a JSON

    all_vfes = np.array(all_vfes)
    all_complexities = np.array(all_complexities)
    all_datafit = np.array(all_datafit)
    all_training_accuracy = np.array(all_training_accuracy)
    all_validation_accuracy = np.array(all_validation_accuracy)

    rescaled_complexities = (all_complexities - np.min(all_complexities)) / (
        np.max(all_complexities) - np.min(all_complexities)
    )
    rescaled_datafit = (all_datafit - np.min(all_datafit)) / (
        np.max(all_datafit) - np.min(all_datafit)
    )

    avg_training_accuracy = np.mean(all_training_accuracy, axis=0)
    std_training_accuracy = np.std(all_training_accuracy, axis=0)
    avg_validation_accuracy = np.mean(all_validation_accuracy, axis=0)
    std_validation_accuracy = np.std(all_validation_accuracy, axis=0)

    # Compute average and standard deviation for complexities and data_fit
    avg_complexities = np.mean(rescaled_complexities, axis=0)
    std_complexities = np.std(rescaled_complexities, axis=0)
    avg_datafit = np.mean(rescaled_datafit, axis=0)
    std_datafit = np.std(rescaled_datafit, axis=0)

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plotting Training and Validation Accuracy with std / avg
    axs[0].fill_between(
        range(epochs),
        avg_training_accuracy - std_training_accuracy,
        avg_training_accuracy + std_training_accuracy,
        alpha=0.2,
    )
    axs[0].fill_between(
        range(epochs),
        avg_validation_accuracy - std_validation_accuracy,
        avg_validation_accuracy + std_validation_accuracy,
        alpha=0.2,
    )
    axs[0].plot(avg_training_accuracy, label="Training Accuracy")
    axs[0].plot(avg_validation_accuracy, label="Validation Accuracy")
    axs[0].set_xscale("log")
    axs[0].set_title("Training and Validation Accuracy")
    axs[0].legend()

    print(f"Std of complexities are: {std_complexities}")
    print(f"Std of datafit are: {std_datafit}")

    # Plotting Complexities and Data Fit with std / avg
    axs[1].fill_between(
        range(epochs),
        avg_complexities - std_complexities,
        avg_complexities + std_complexities,
        alpha=0.2,
    )
    axs[1].fill_between(
        range(epochs), avg_datafit - std_datafit, avg_datafit + std_datafit, alpha=0.2
    )
    axs[1].plot(avg_complexities, label="Normalised Complexity")
    axs[1].plot(avg_datafit, label="Normalised Data Fit")
    axs[1].set_xscale("log")
    axs[1].set_title("Complexity and Data Fit (Normalised)")
    axs[1].legend()

    # Saving to PDF
    plt.tight_layout()
    plt.savefig(
        "experiments/grokking_accessibility_vafe/results.pdf",
        format="pdf",
        bbox_inches="tight",
    )


def experiment_spheres_around_optimal_solution():
    """
    Assumes that experiment grokking_accessibility_vafe has been run and its result
    is available.
    """

    experiment_path = Path("experiments/spheres_around_optimal_solution/")

    os.makedirs(experiment_path, exist_ok=True)

    for n_directions in [5, 7, 10, 15, 20]:
        top_n_directions = n_directions
        radii = [
            5e-2,
            1e-1,
            1.2e-1,
            1.5e-1,
            1.8e-1,
            2e-1,
            3e-1,
            4e-1,
            5e-1,
            7e-1,
            1e0,
            1.5e0,
            2e0,
            2.5e0,
            3e0,
        ]

        batch_size = 32
        input_size = 30
        k = 3
        number_training_samples = 600
        number_validation_samples = 200

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        random_seed = 0

        # Load PCA directions
        try:
            with open(
                "experiments/grokking_accessibility_vafe/top_20_pca_directions.json",
                "r",
            ) as json_file:
                top_20_directions = json.load(json_file)
        except FileNotFoundError:
            print(
                "Please run experiment_grokking_accessibility_vafe first to generate the top 20 directions"
            )
            return

        # Load the model
        try:
            with open(
                "experiments/grokking_accessibility_vafe/final_model.pkl", "rb"
            ) as f:
                model = pickle.load(f)
        except FileNotFoundError:
            print(
                "Please run experiment_grokking_accessibility_vafe first to generate the final model"
            )
            return

        loss_function = torch.nn.CrossEntropyLoss()

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

        top_20_directions = np.array(top_20_directions)

        complexity_with_all_radii = []
        error_with_all_radii = []
        accuracy_with_all_radii = []
        validation_accuracy_with_all_radii = []

        original_model = copy.deepcopy(model)

        for radius in radii:
            complexity_with_radius = []
            error_with_radius = []
            accuracy_with_radius = []
            validation_accuracy_with_radius = []

            for random_seed in tqdm(range(0, 50), desc=f"Sampling for radius {radius}"):
                model = original_model

                additional_weight = sample_in_subspace(
                    top_20_directions[:top_n_directions], radius, random_seed
                )

                # Update the weights of model to be w - radius * top_n_directions

                layer_1_weights = model.layers[0].q_mean.cpu().detach().numpy()
                layer_2_weights = model.layers[1].q_mean.cpu().detach().numpy()

                layer_1_vector = additional_weight[
                    : layer_1_weights.shape[0] * layer_1_weights.shape[1]
                ]
                layer_2_vector = additional_weight[
                    layer_1_weights.shape[0] * layer_1_weights.shape[1] :
                ]

                layer_1_matrix = layer_1_vector.reshape(layer_1_weights.shape)
                layer_2_matrix = layer_2_vector.reshape(layer_2_weights.shape)

                parameter_for_layer_1 = torch.nn.Parameter(
                    torch.tensor(
                        layer_1_weights - layer_1_matrix, dtype=torch.float32
                    ).to(device)
                )

                model.layers[0].q_mean = parameter_for_layer_1

                model.layers[1].q_mean = torch.nn.Parameter(
                    torch.tensor(
                        layer_2_weights - layer_2_matrix, dtype=torch.float32
                    ).to(device)
                )

                number_batches = (
                    total_vafe
                ) = (
                    total_error
                ) = total_complexity = total_accuracy = total_validation_accuracy = 0

                for batch in train_loader:
                    inputs = batch[0].contiguous().to(device, non_blocking=False)
                    targets = batch[1].contiguous().to(device, non_blocking=False)

                    predictions = model(inputs)

                    loss_error = loss_function(predictions, targets)
                    complexity_loss = model.kl() / len(train_loader.dataset)
                    loss = loss_error + complexity_loss
                    # loss = loss_error

                    total_vafe += loss_error.item() + complexity_loss.item()
                    total_complexity += complexity_loss.item()

                    accuracy = get_accuracy(predictions, targets)

                    total_accuracy += accuracy

                    number_batches += 1

                total_vafe /= number_batches
                total_complexity /= number_batches
                total_error = total_vafe - total_complexity

                total_accuracy /= number_batches

                complexity_with_radius.append(total_complexity)
                error_with_radius.append(total_error)
                accuracy_with_radius.append(total_accuracy)

                number_batches = 0

                for batch in validation_loader:
                    inputs = batch[0].contiguous().to(device, non_blocking=False)
                    targets = batch[1].contiguous().to(device, non_blocking=False)

                    predictions = model(inputs)

                    accuracy = get_accuracy(predictions, targets)

                    total_validation_accuracy += accuracy

                    number_batches += 1

                total_validation_accuracy /= number_batches
                validation_accuracy_with_radius.append(total_validation_accuracy)

            complexity_with_radius = np.array(complexity_with_radius)
            error_with_radius = np.array(error_with_radius)
            accuracy_with_radius = np.array(accuracy_with_radius)
            validation_accuracy_with_radius = np.array(validation_accuracy_with_radius)

            complexity_with_all_radii.append(complexity_with_radius)
            error_with_all_radii.append(error_with_radius)
            accuracy_with_all_radii.append(accuracy_with_radius)
            validation_accuracy_with_all_radii.append(validation_accuracy_with_radius)

        complexity_with_all_radii = np.array(complexity_with_all_radii)
        error_with_all_radii = np.array(error_with_all_radii)
        accuracy_with_all_radii = np.array(accuracy_with_all_radii)
        validation_accuracy_with_all_radii = np.array(
            validation_accuracy_with_all_radii
        )

        plot_combined_relationship(
            radii,
            [
                complexity_with_all_radii,
                error_with_all_radii,
                accuracy_with_all_radii,
                validation_accuracy_with_all_radii,
            ],
            ["Complexity", "Error", "Accuracy", "Validation Accuracy"],
            "Metrics with different radii",
            experiment_path / Path(f"combined_metrics_{n_directions}.pdf"),
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
        "--experiments", type=str, nargs="+", default=[], help="Experiments to run"
    )

    args = argparser.parse_args()

    for experiment_name in args.experiments:
        os.makedirs("experiments/experiment_name/", exist_ok=True)
        eval("experiment_{}()".format(experiment_name))
