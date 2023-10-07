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
from pathlib import Path
import os

from src.dataset import (
    ParityTask,
    HiddenDataset,
    PeekParityTask,
    ModuloAdditionTask,
    combine_datasets,
)
from src.model import TinyModel, ExpandableModel, TinyBayes
from src.train import train_model, Observer
from src.plot import (
    plot_validation_and_accuracy_from_observers,
)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
from tqdm import tqdm


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

    weight_norms = [1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2]

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
    epochs = 4000
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


def experiment_training_with_vafe(args):
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


def experiment_grokking_plain_with_vafe(args):
    """
    We look at the variational free energy and its components during grokking.
    """

    os.makedirs("experiments/grokking_plain_with_vafe/", exist_ok=True)

    weight_decay = 0
    learning_rate = 1e-1
    batch_size = 32
    input_size = 40
    output_size = 2
    k = 3
    hidden_size = 200
    epochs = 650
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
        q_mean_std=0.1,
    )

    observer = Observer(
        observation_settings={"variational_free_energy": {}},
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

    observer.plot_me(path=Path("experiments/grokking_plain_with_vafe/"), log=True)


def experiment_complexity_error_bnn_grokking(args):
    """
    In this experiment we aim to produce a graph where on one axis there is complexity and the other there is error. Then we plot various trajectories and cover them based on the amount of grokking.
    """

    os.makedirs("experiments/complexity_error_bnn_grokking/", exist_ok=True)

    weight_decay = 0
    learning_rate = 1e-1
    batch_size = 32
    input_size = 30
    output_size = 2
    k = 3
    hidden_size = 200
    epochs = 1500
    number_training_samples = 1000
    number_validation_samples = 200
    random_seed = 2

    threshold = 0.95

    datafit_list = []
    complexity_list = []
    grokking_gap_list = []
    training_accuracies = []
    validation_accuracies = []
    complexities_to_grok = []
    training_errors_to_grok = []

    variances = [5e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 7e-1, 1e0]

    for variance in tqdm(variances):
        cache_file = (
            f"experiments/complexity_error_bnn_grokking/results_{variance}_{epochs}_seed{random_seed}.pkl"
        )

        # Check if results for the current variance exist in cache
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                cached_results = pickle.load(f)
            datafit = np.array(cached_results["datafit"])
            complexity = np.array(cached_results["complexity"])

            training_accuracy = np.array(cached_results["training_accuracy"])
            validation_accuracy = np.array(cached_results["validation_accuracy"])

        else:
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
                q_mean_std=variance,
            )

            observer = Observer(
                observation_settings={"variational_free_energy": {}},
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

            training_accuracy = np.array(observer.training_accuracy)
            validation_accuracy = np.array(observer.validation_accuracy)

            datafit = np.array(observer.error_loss)
            complexity = np.array(observer.complexity_loss)

        print(f"Training accuracy for variance {variance}: {training_accuracy}")
        print(f"Validation accuracy for variance {variance}: {validation_accuracy}")

        # Find the point where training accuracy is above threshold
        training_indices_above_95 = np.where(training_accuracy > threshold)[0]
        if training_indices_above_95.size > 0:
            training_accuracy_above_95 = training_indices_above_95[0]
        else:
            training_accuracy_above_95 = (
                epochs  # Set to total number of epochs if never exceeds threshold
            )

        validation_indices_above_95 = np.where(validation_accuracy > threshold)[0]
        if validation_indices_above_95.size > 0:
            validation_accuracy_above_95 = validation_indices_above_95[0]
        else:
            validation_accuracy_above_95 = (
                epochs  # Set to total number of epochs if never exceeds threshold
            )

        print(f"Validation accuracy 95: {validation_accuracy_above_95}")

        grokking_gap = validation_accuracy_above_95 - training_accuracy_above_95

        print(f"Grokking gap for variance {variance}: {grokking_gap}")

        if validation_accuracy_above_95 != epochs:
            complexities_to_grok.append(complexity[validation_accuracy_above_95])
            training_errors_to_grok.append(datafit[validation_accuracy_above_95])

        # Store the values in lists
        datafit_list.append(datafit)
        complexity_list.append(complexity)
        grokking_gap_list.append(grokking_gap)

        results_to_cache = {
            "datafit": datafit,
            "complexity": complexity,
            "grokking_gap": grokking_gap,
            "training_accuracy": training_accuracy,
            "validation_accuracy": validation_accuracy,
        }

        with open(cache_file, "wb") as f:
            pickle.dump(results_to_cache, f)

    if complexities_to_grok != []:
        print(f"Complexities to grok: {complexities_to_grok}")

        average_complexity_to_grok = np.mean(complexities_to_grok)
        print(f"Average complexity to grok: {average_complexity_to_grok}")

    if training_errors_to_grok != []:
        print(f"Training errors to grok: {training_errors_to_grok}")

        average_training_error_to_grok = np.mean(training_errors_to_grok)
        print(f"Average training error to grok: {average_training_error_to_grok}")

    print(f"Grokking gap list: {grokking_gap_list}")

    # After the loop ends, plot the values
    # Create a new figure with a width of 12 units and height of 5 units
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)

    normalized_grokking_gap = [
        (gap - min(grokking_gap_list))
        / (max(grokking_gap_list) - min(grokking_gap_list))
        for gap in grokking_gap_list
    ]

    print(f"Normalised grokking gap: {normalized_grokking_gap}")

    cmap = cm.coolwarm

    # Define max and min alpha values
    max_alpha = 1.0
    min_alpha = 0.3

    for i in range(3, len(datafit_list)):
        if complexities_to_grok != []:
            print(
                f"Grokking point is: {training_errors_to_grok[i]}, {complexities_to_grok[i]}"
            )

            if i == 3:
                label = "Grokking Point"

                plt.plot(
                    10000,
                    10000,
                    "^",
                    color="black",
                    markersize=10,
                    label=label,
                    alpha=1.0,
                    markerfacecolor="white",
                    markeredgecolor="black",
                    zorder=100000,
                )
            else:
                label = ""

            plt.plot(
                training_errors_to_grok[i],
                complexities_to_grok[i],
                "^",
                color="black",
                markersize=10,
                label="",
                alpha=1.0,
                markerfacecolor=cmap(normalized_grokking_gap[i]),
                markeredgecolor="black",
                zorder=100000,
            )

        num_points = len(datafit_list[i])
        for j in range(num_points):
            alpha_max = 1.0
            alpha_min = 0.2
            k = -np.log(0.1 / 0.9) / 100

            alpha_val = alpha_min + (alpha_max - alpha_min) * np.exp(-k * j)

            plt.plot(
                datafit_list[i][j],
                complexity_list[i][j],
                "o",
                label=f"$\sigma_{i-3} = {variances[i]}$" if j == 0 else "",
                color=cmap(normalized_grokking_gap[i]),
                linewidth=3.5,
                alpha=alpha_val,  # Set alpha based on the logarithmic index within the sublist
            )
        # Connect points with a line, without changing alpha for lines
        # plt.plot(datafit_list[i], complexity_list[i], "-", color=cmap(normalized_grokking_gap[i]), linewidth=3.5)

        # Plot the location of the point where the model groks

    # plt.axhline(
    #     y=average_complexity_to_grok,
    #     color="black",
    #     linestyle="--",
    #     label="Complextiy to Grok",
    # )

    # plt.axvline(
    #     x=average_training_error_to_grok,
    #     color="black",
    #     linestyle="--",
    #     label="Error to Grok",
    # )

    plt.xlabel("Error")
    plt.ylabel("Complexity")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim([10, 3.2e1])
    plt.xlim([5e-3, 30])
    plt.title("Error and Complexity")
    plt.legend()

    # Link the colorbar to the normalized colormap
    sm = cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, label="Normalised Grokking Gap")

    # Second subplot: the new figure
    plt.subplot(1, 2, 2)

    average_datafit = np.mean(datafit_list, axis=0)
    average_complexity = np.mean(complexity_list, axis=0)

    std_datafit = np.std(datafit_list, axis=0)
    std_complexity = np.std(complexity_list, axis=0)

    plt.plot(average_datafit, label="Average Error")
    plt.plot(average_complexity, label="Average Complexity")
    plt.fill_between(
        range(len(average_datafit)),
        average_datafit - std_datafit,
        average_datafit + std_datafit,
        alpha=0.3,
    )
    plt.fill_between(
        range(len(average_complexity)),
        average_complexity - std_complexity,
        average_complexity + std_complexity,
        alpha=0.3,
    )

    plt.axhline(
        y=average_complexity_to_grok,
        color="black",
        linestyle="--",
        label="Average Complextiy Required to Grok",
    )

    plt.xlabel("Epochs")
    plt.xscale("log")
    plt.ylabel("Value")
    plt.title("Average Error and Complexity")
    plt.legend()

    # Save the combined figure
    plt.tight_layout()  # This will ensure that the subplots do not overlap
    plt.savefig(
        f"experiments/complexity_error_bnn_grokking/combined_figure_{epochs}_seed{random_seed}.pdf",
        bbox_inches="tight",
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
        "--experiments", type=str, nargs="+", default=["complexity_error_bnn_grokking"], help="Experiments to run"
    )

    args = argparser.parse_args()

    for experiment_name in args.experiments:
        os.makedirs("experiments/experiment_name/", exist_ok=True)
        eval("experiment_{}(args)".format(experiment_name))