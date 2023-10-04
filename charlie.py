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
    entire_dataset = ParityTask(
        sequence_length=args.k_factor_range,
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
        [args.num_samples - args.validation_samples, args.validation_samples],
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
        loss_function_label=args.loss_function_label,
        optimiser_function_label=args.optimiser_label,
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
    # Create a new figure with a width of 12 units and height of 6 units
    plt.figure(figsize=(12, 6))
    
    # First subplot: Error and Complexity
    plt.subplot(1, 2, 1)

    normalized_grokking_gap = [
        (gap - min(grokking_gap_list))
        / (max(grokking_gap_list) - min(grokking_gap_list))
        for gap in grokking_gap_list
    ]

    cmap = cm.coolwarm

    for i, (datafit, complexity) in enumerate(zip(datafit_list, complexity_list)):
        alpha_max = 1.0
        alpha_min = 0.2
        k = -np.log(0.1 / 0.9) / 100

        num_points = len(datafit)
        for j in range(num_points):
            alpha_val = alpha_min + (alpha_max - alpha_min) * np.exp(-k * j)
            plt.plot(
                datafit[j],
                complexity[j],
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
    plt.title("Error and Complexity")
    plt.legend()

    sm = cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, label="Normalized Grokking Gap")

    # Second subplot: Average Error and Complexity
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

    # Save the combined figure
    plt.tight_layout()
    plt.savefig(
        f"experiments/{args.experiment_name}/combined_figure_{args.epochs}.pdf",
        bbox_inches="tight",
    )

def main(args):
    os.makedirs(f"experiments/{args.experiment_name}/", exist_ok=True)
    random_seed = 0
    threshold = 0.95

    training_dataset, validation_dataset = prepare_data(args, random_seed)
    datafit_list, complexity_list, grokking_gap_list = [], [], []

    for variance in tqdm(args.variances):
        cache_file = f"experiments/{args.experiment_name}/results_{variance}_{args.epochs}.pkl"
        if os.path.exists(cache_file):
            cached_results = load_from_cache(cache_file)
        else:
            model, observer = train_and_evaluate_model(args, training_dataset, validation_dataset, variance, random_seed)
            cached_results = {
                "datafit": np.array(observer.error_loss),
                "complexity": np.array(observer.complexity_loss),
                "grokking_gap": calculate_grokking_metrics(np.array(observer.training_accuracy), np.array(observer.validation_accuracy), threshold, args.epochs)
            }
            save_to_cache(cached_results, cache_file)

        datafit_list.append(cached_results["datafit"])
        complexity_list.append(cached_results["complexity"])
        grokking_gap_list.append(cached_results["grokking_gap"])

    plot_results(args, datafit_list, complexity_list, grokking_gap_list)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    # Populate argparse arguments here
    args = argparser.parse_args()
    main(args)