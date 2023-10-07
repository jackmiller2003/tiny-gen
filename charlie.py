import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import torch
import matplotlib.cm as cm
from src.dataset import ParityTask, HiddenDataset
from src.model import TinyBayes
from src.train import train_model, Observer
from src.plot import plot_validation_and_accuracy_from_observers

# Additional Helper Functions
def save_to_cache(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load_from_cache(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

# Modular Functions
def prepare_data(args, random_seed):
    validation_samples = 200
    entire_dataset = ParityTask(
        sequence_length=3,  
        num_samples=args.num_samples,
        random_seed=random_seed,
    )
    hidden_dataset = HiddenDataset(
        dataset=entire_dataset,
        total_length=args.sequence_length,
        random_seed=random_seed,
    )
    return torch.utils.data.random_split(
        hidden_dataset,
        [args.num_samples - validation_samples, validation_samples],
    )

def train_and_evaluate_model(args, training_dataset, validation_dataset, variance, random_seed):
    model = TinyBayes(
        input_size=args.sequence_length,
        hidden_layer_size=args.hidden_layer_size,
        output_size=2,
        random_seed=random_seed,
        normalise_output=False,
        q_mean_std=variance,
    )
    observer = Observer(observation_settings={"variational_free_energy": {}})
    return train_model(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        loss_function_label="cross-entropy",
        optimiser_function_label="sgd",
        progress_bar=True,
        observer=observer,
    )

def calculate_grokking_metrics(training_accuracy, validation_accuracy, threshold, epochs):
    training_indices_above_thresh = np.where(training_accuracy > threshold)[0]
    training_accuracy_above_thresh = training_indices_above_thresh[0] if training_indices_above_thresh.size > 0 else epochs
    validation_indices_above_thresh = np.where(validation_accuracy > threshold)[0]
    validation_accuracy_above_thresh = validation_indices_above_thresh[0] if validation_indices_above_thresh.size > 0 else epochs
    return validation_accuracy_above_thresh - training_accuracy_above_thresh

def plot_results(args, datafit_list, complexity_list, grokking_gap_list):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)

    normalized_grokking_gap = [
        (gap - min(grokking_gap_list))
        / (max(grokking_gap_list) - min(grokking_gap_list))
        for gap in grokking_gap_list
    ]

    cmap = cm.coolwarm

    for i in range(len(datafit_list)):
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
                label=f"$\sigma_{i} = {args.variances[i]}$" if j == 0 else "",
                color=cmap(normalized_grokking_gap[i]),
                linewidth=3.5,
                alpha=alpha_val,
            )

    plt.xlabel("Error")
    plt.ylabel("Complexity")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim([0.1, 1])
    plt.xlim([0.1, 1])
    plt.title("Error and Complexity")
    plt.legend()

    sm = cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, label="Normalized Grokking Gap")

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

    plt.xlabel("Epochs")
    plt.xscale("log")
    plt.ylabel("Value")
    plt.title("Average Error and Complexity")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"experiments/{args.experiment_name}/combined_figure_{args.epochs}.pdf", bbox_inches="tight")

def main(args):
    os.makedirs(f"experiments/{args.experiment_name}/", exist_ok=True)
    random_seed = 0  # Initialize random seed
    threshold = 0.95  # Initialize threshold for grokking

    num_runs = 3  # Number of runs per variance

    # Initialize lists to store the results
    datafit_list_runs = []
    complexity_list_runs = []
    grokking_gap_list_runs = []

    for run in tqdm(range(num_runs), desc="Total runs"):
        # Re-initialize per-run lists
        datafit_list = []
        complexity_list = []
        grokking_gap_list = []

        training_dataset, validation_dataset = prepare_data(args, random_seed + run)

        for variance in tqdm(args.variances):
            cache_file = f"experiments/{args.experiment_name}/results_run{run}_{variance}_{args.epochs}.pkl"

            if os.path.exists(cache_file):
                print(f"Loading cached results from {cache_file}")
                cached_results = load_from_cache(cache_file)
            else:
                print(f"Couldn't find cached {cache_file}, training from scratch")
                model, observer = train_and_evaluate_model(
                    args, training_dataset, validation_dataset, variance, random_seed + run
                )
                grokking_gap = calculate_grokking_metrics(np.array(observer.training_accuracy), 
                                                          np.array(observer.validation_accuracy), 
                                                          threshold, args.epochs)
                cached_results = {
                    "datafit": np.array(observer.error_loss),
                    "complexity": np.array(observer.complexity_loss),
                    "grokking_gap": grokking_gap,
                    "training_accuracy": observer.training_accuracy,
                    "validation_accuracy": observer.validation_accuracy,
                }
                save_to_cache(cached_results, cache_file)

            datafit_list.append(cached_results["datafit"])
            complexity_list.append(cached_results["complexity"])
            grokking_gap_list.append(cached_results["grokking_gap"])

        # Append per-run lists to the overall lists
        datafit_list_runs.append(datafit_list)
        complexity_list_runs.append(complexity_list)
        grokking_gap_list_runs.append(grokking_gap_list)

    # Average the runs
    avg_datafit_list = np.mean(datafit_list_runs, axis=0)
    avg_complexity_list = np.mean(complexity_list_runs, axis=0)
    avg_grokking_gap_list = np.mean(grokking_gap_list_runs, axis=0)

    plot_results(args, avg_datafit_list, avg_complexity_list, avg_grokking_gap_list)

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
        default=200,
        help="Size of the hidden layer in the network",
    )

    argparser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-1,
        help="Learning rate for the network",
    )

    argparser.add_argument(
        "--weight_decay", type=float, default=0, help="Weight decay for the network"
    )

    argparser.add_argument(
        "--epochs", type=int, default=1500, help="Number of epochs to train for"
    )

    argparser.add_argument(
        "--batch_size", type=int, default=4096, help="Batch size for training"
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
        "--experiment_name",
        type=str,
        default="complexity_error_bnn_grokking",
        help="Name of the experiment",
    )

    argparser.add_argument(
        "--variances",
        type=float,
        nargs='+',
        default=[5e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 7e-1, 1e0],
        help="List of variances to use for the model",
    )

    args = argparser.parse_args()
    main(args)