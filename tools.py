from torch.utils.data import Dataset
import torch
from typing import Any, Optional
from pathlib import Path
from tqdm import tqdm
from src.model import ExactMarginalLikelihood
from matplotlib import pyplot as plt
import copy
import numpy as np
from src.train import setup_optimiser_and_loss
import matplotlib.gridspec as gridspec


def plot_landsacpes_of_GP_model(
    training_dataset: Dataset,
    model: Any,
    likelihood: Any,
    path_to_plot: Path,
    num_plotting_steps: int,
    loss_function_label: Optional[str] = "mse",
    optimiser_function_label: Optional[str] = "sgd",
    epochs: Optional[int] = 1000,
    learning_rate: Optional[float] = 4e-2,
    validation_dataset: Optional[Dataset] = None,
    trajectories_through_landscape: Optional[bool] = False,
    progress_bar: Optional[bool] = True,
) -> None:
    """
    Plots the marginal likelihood landscape, datafit landscape and complexity landscape.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_inputs = torch.tensor(
        [x.clone().detach().unsqueeze(0) for x, y in training_dataset]
    ).to(device)

    train_targets = torch.tensor(
        [y.clone().detach().unsqueeze(0) for x, y in training_dataset]
    ).to(device)

    if validation_dataset is not None:
        validation_inputs = torch.tensor(
            [x.clone().detach().unsqueeze(0) for x, y in validation_dataset]
        ).to(device)

        validation_targets = torch.tensor(
            [y.clone().detach().unsqueeze(0) for x, y in validation_dataset]
        ).to(device)

    log_scale = torch.linspace(-5, 5, num_plotting_steps)
    log_lengthscale = torch.linspace(-7, 4, num_plotting_steps)
    ln, ll = torch.meshgrid(log_scale, log_lengthscale, indexing="xy")
    ml = torch.zeros_like(ln)
    fit_terms = torch.zeros_like(ln)
    complexity_terms = torch.zeros_like(ln)
    for i in tqdm(range(ln.shape[0]), disable=not progress_bar):
        for j in range(ln.shape[1]):
            mll = ExactMarginalLikelihood(likelihood, model)
            model.to(device)
            likelihood.to(device)
            model.covar_module.base_kernel.lengthscale = float(torch.exp(ll[i, j]))
            model.covar_module.outputscale = float(torch.exp(ln[i, j]))

            output = model(train_inputs)
            loss, fit, comp = mll(output, train_targets)
            loss, fit, comp = -loss, -fit, -comp
            ml[i, j] = loss.detach().cpu()
            fit_terms[i, j] = fit.detach().cpu()
            complexity_terms[i, j] = comp.detach().cpu()

    plt.figure(1)
    c = plt.pcolor(ln, ll, ml, cmap="RdBu")
    plt.colorbar(c)
    plt.xlabel("log outputscale")
    plt.ylabel("log lengthscale")
    plt.savefig(path_to_plot / Path("gpr_marginal_likelihood_landscape.pdf"))

    plt.figure(2)
    c = plt.pcolor(ln, ll, fit_terms, cmap="RdBu")
    plt.colorbar(c)
    plt.xlabel("log outputscale")
    plt.ylabel("log lengthscale")
    plt.savefig(path_to_plot / Path("gpr_data_fit_landscape.pdf"))

    plt.figure(3)
    c = plt.pcolor(ln, ll, complexity_terms, cmap="RdBu")
    plt.colorbar(c)
    plt.xlabel("log outputscale")
    plt.ylabel("log lengthscale")
    plt.savefig(path_to_plot / Path("gpr_complexity_landscape.pdf"))

    likelihood_init = copy.deepcopy(likelihood)
    model_init = copy.deepcopy(model)

    if trajectories_through_landscape:
        init_ll = [-6, -6, 0]
        init_ls = [-4, 4, 2]
        colors = ["g", "y", "w"]
        ll_paths = []
        ls_paths = []
        all_train_mses = []
        all_val_mses = []

        for init_idx, (ll_i, ls_i) in tqdm(enumerate(zip(init_ll, init_ls))):
            likelihood = copy.deepcopy(likelihood_init)
            model = copy.deepcopy(model_init)
            mll = ExactMarginalLikelihood(likelihood, model)
            model.to(device)
            likelihood.to(device)

            loss_function, optimiser = setup_optimiser_and_loss(
                loss_function_label=loss_function_label,
                optimiser_function_label=optimiser_function_label,
                learning_rate=learning_rate,
                model=model,
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
                optimiser.zero_grad()
                output = model(train_inputs)
                loss, fit, comp = mll(output, train_targets)
                loss, fit, comp = -loss, -fit, -comp
                loss.backward()

                model.eval()
                output = model(train_inputs)
                train_preds = likelihood(output)
                valid_output = model(validation_inputs)
                valid_preds = likelihood(valid_output)
                model.train()

                train_loss = (
                    loss_function(output.mean, train_targets).mean().detach().cpu()
                )
                valid_loss = (
                    loss_function(valid_output.mean, validation_targets)
                    .mean()
                    .detach()
                    .cpu()
                )
                train_lp = train_preds.log_prob(train_targets).mean().detach().cpu()
                valid_lp = (
                    valid_preds.log_prob(validation_targets).mean().detach().cpu()
                )
                train_mses.append(train_loss)
                train_lps.append(train_lp)
                valid_mses.append(valid_loss)
                valid_lps.append(valid_lp)
                lmls.append(loss.detach().cpu())
                fits.append(fit.detach().cpu())
                comps.append(comp.detach().cpu())

                # if i in epochs_ls:
                #     lengthscales.append(
                #         model.covar_module.base_kernel.lengthscale.detach()
                #         .cpu()
                #         .numpy()
                #     )
                #     model.eval()
                #     x = torch.from_numpy(np.linspace(-2, 2, 100)).to(device)
                #     plt.figure()
                #     plt.plot(train_inputs.cpu(), train_targets.cpu(), "+k")
                #     pred = model(x)
                #     m, v = pred.mean.detach(), pred.variance.detach()
                #     plt.plot(x.cpu(), m.cpu(), "-b")
                #     plt.fill_between(
                #         x.cpu(),
                #         m.cpu() + 2 * torch.sqrt(v.cpu()),
                #         m.cpu() - 2 * torch.sqrt(v.cpu()),
                #         color="b",
                #         alpha=0.3,
                #     )
                #     plt.ylim(-3, 3)

                #     plt.savefig(path_to_plot / Path(f"gpr_{init_idx}_pred_{i}.pdf"))

                optimiser.step()

            # --- Adding trajectories to existing landscape --- #

            ll_paths.append(ll_path)
            ls_paths.append(ls_path)
            all_train_mses.append(train_mses)
            all_val_mses.append(valid_mses)

            plt.figure(1)
            plt.plot(ls_path, ll_path, color="k")
            plt.plot(ls_path[0], ll_path[0], "s", markersize=10, color=colors[init_idx])
            plt.plot(
                ls_path[-1], ll_path[-1], "*", markersize=10, color=colors[init_idx]
            )
            plt.title("Trajectories through marginal likelihood landscape")
            plt.savefig(
                path_to_plot / Path("gpr_marginal_likelihood_landscape_trajectory.pdf")
            )

            plt.figure(2)
            plt.title("Trajectories through data fit landscape")
            plt.plot(ls_path, ll_path, color="k")
            plt.plot(ls_path[0], ll_path[0], "s", markersize=10, color=colors[init_idx])
            plt.plot(
                ls_path[-1], ll_path[-1], "*", markersize=10, color=colors[init_idx]
            )
            plt.savefig(path_to_plot / Path("gpr_data_fit_landscape_trajectory.pdf"))

            plt.figure(3)
            plt.title("Trajectories through complexity landscape")
            plt.plot(ls_path, ll_path, color="k")
            plt.plot(ls_path[0], ll_path[0], "s", markersize=10, color=colors[init_idx])
            plt.plot(
                ls_path[-1], ll_path[-1], "*", markersize=10, color=colors[init_idx]
            )
            plt.savefig(path_to_plot / Path("gpr_complexity_landscape_trajectory.pdf"))

            plt.figure()
            plt.plot(np.arange(epochs) + 1, train_mses, "-r", label="train")
            plt.plot(np.arange(epochs) + 1, valid_mses, "-b", label="validation")
            plt.xscale("log")
            plt.xlabel("epoch")
            plt.ylabel("mse")
            plt.legend()
            plt.savefig(path_to_plot / Path(f"gpr_{init_idx}_mse.pdf"))

            plt.figure()
            plt.plot(np.arange(epochs) + 1, train_lps, "-r", label="train")
            plt.plot(np.arange(epochs) + 1, valid_lps, "-b", label="validation")
            plt.xscale("log")
            plt.xlabel("epoch")
            plt.ylabel("log prob")
            plt.legend()
            plt.savefig(path_to_plot / Path(f"gpr_{init_idx}_lp.pdf"))

            plt.figure()
            plt.plot(np.arange(epochs) + 1, lmls, "-k")
            plt.plot(np.arange(epochs) + 1, fits, "-b")
            plt.plot(np.arange(epochs) + 1, comps, "-r")
            plt.xscale("log")
            plt.xlabel("epoch")
            plt.ylabel("objective, data fit and complexity")
            plt.savefig(path_to_plot / Path(f"gpr_{init_idx}_lml.pdf"))

        # Create a gridspec layout for the figure
        gs = gridspec.GridSpec(3, 3, height_ratios=[1, 0.05, 1], hspace=0.5)

        fig = plt.figure(figsize=(18, 12))

        # Create the main plots using the gridspec
        axs = [
            [fig.add_subplot(gs[0, i]) for i in range(3)],
            [fig.add_subplot(gs[2, i]) for i in range(3)],
        ]

        # Create axes for colorbars below each of the top row subplots
        caxs = [fig.add_subplot(gs[1, i]) for i in range(3)]

        # Plot all trajectories
        for init_idx in range(0, 3):
            for i in range(0, 3):
                axs[0][i].plot(
                    ls_paths[init_idx], ll_paths[init_idx], color="k", zorder=1
                )
                axs[0][i].plot(
                    ls_paths[init_idx][0],
                    ll_paths[init_idx][0],
                    "s",
                    markersize=15,
                    color="w",
                    zorder=200,
                )

                if init_idx == 0:
                    axs[0][i].plot(
                        ls_paths[init_idx][-1],
                        ll_paths[init_idx][-1],
                        "o",
                        markersize=15,
                        color="w",
                        zorder=200,
                    )

                axs[0][i].text(
                    ls_paths[init_idx][0],
                    ll_paths[init_idx][0],
                    chr(65 + init_idx),
                    color="k",
                    ha="center",  # horizontal alignment
                    va="center",
                    zorder=300,
                )
                label_offset = 0.01  # adjust this value for the best visual appearance

            axs[1][init_idx].plot(
                np.arange(epochs) + 1, all_train_mses[init_idx], "-r", label="train"
            )
            axs[1][init_idx].plot(
                np.arange(epochs) + 1, all_val_mses[init_idx], "-b", label="validation"
            )
            axs[1][init_idx].set_xscale("log")
            axs[1][init_idx].set_xlabel("Epoch")
            axs[1][init_idx].set_ylabel("Mean squared error")
            axs[1][init_idx].set_title(f"MSE for initialisation {chr(65 + init_idx)}")
            if init_idx == 0:
                axs[1][init_idx].legend()

        # Marginal likelihood landscape
        c1 = axs[0][0].pcolor(ln, ll, ml, cmap="RdBu")
        fig.colorbar(c1, cax=caxs[0], orientation="horizontal")
        axs[0][0].set_xlabel("log outputscale")
        axs[0][0].set_ylabel("log lengthscale")
        axs[0][0].plot(ls_path, ll_path, color="k")
        axs[0][0].set_title("Trajectories through marginal likelihood landscape")

        # Data fit landscape
        c2 = axs[0][1].pcolor(ln, ll, fit_terms, cmap="RdBu")
        fig.colorbar(c2, cax=caxs[1], orientation="horizontal")
        axs[0][1].set_xlabel("log outputscale")
        axs[0][1].set_ylabel("log lengthscale")
        axs[0][1].set_title("Trajectories through data fit landscape")

        # Complexity landscape
        c3 = axs[0][2].pcolor(ln, ll, complexity_terms, cmap="RdBu")
        fig.colorbar(c3, cax=caxs[2], orientation="horizontal")
        axs[0][2].set_xlabel("log outputscale")
        axs[0][2].set_ylabel("log lengthscale")
        axs[0][2].set_title("Trajectories through complexity landscape")

        # Save the combined figure
        plt.tight_layout()  # Adjust the spacing between subplots for better appearance
        plt.savefig(
            path_to_plot / Path("combined_landscape_trajectory.pdf"),
            bbox_inches="tight",
        )


# we add spurious features
def add_features_for_lr_classification(x):
    fs = [lambda x: x**2, lambda x: x**3, lambda x: torch.sin(100 * x)]
    for f in fs:
        x = torch.cat([x, f(x[:, 0]).unsqueeze(-1)], 1)
    return x


def gaussian_loss(y_pred, y, noise_var=0.002):
    loss = torch.nn.functional.mse_loss(y_pred, y, reduction="sum")
    loss /= noise_var
    return loss


def l2_norm_for_lr(
    model,
    device,
    prior_mean=torch.tensor([[0, 0, 0, 0]]),
    prior_var=torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
):
    norm = (model.fc1.weight - prior_mean.to(device)) ** 2 / prior_var.to(device)
    return norm.sum()


def accuracy_for_negative_positive(y_pred, y):
    """
    if y_pred is above 0 and y is above 0, that's correct, and vice versa.
    """
    # Convert to binary labels: 1 for >0 and 0 for <=0
    pred_labels = (y_pred > 0).float()
    true_labels = (y > 0).float()

    # Count correct predictions
    correct = (pred_labels == true_labels).float().sum()

    # Compute accuracy
    acc = correct / y_pred.size(0)

    return acc.item()


def plot_lr_pred(epoch, model, x_plot, x_valid, y_valid, x_train, y_train):
    y_pred = model(x_plot).squeeze().cpu().detach().numpy()

    plt.figure()
    plt.scatter(
        x_valid[:, 0].cpu().numpy(),
        y_valid.cpu().numpy(),
        color="red",
        s=50,
        label="Validation Data",
    )
    plt.scatter(
        x_train[:, 0].cpu().numpy(),
        y_train.cpu().numpy(),
        color="blue",
        s=50,
        label="Train Data",
    )
    plt.plot(x_plot[:, 0].cpu().numpy(), y_pred, color="green", label="Prediction")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("linear regression with spurious features")
    plt.legend()

    plt.ylim([-1, 1])

    plt.savefig("tmp/grokking_with_lr_inference_%d.png" % epoch)


def get_rows_for_dataset(final_array, dataset_index, random_seeds):
    rows_per_dataset = len(random_seeds)
    start_index = dataset_index * rows_per_dataset
    end_index = start_index + rows_per_dataset
    return final_array[start_index:end_index]


def sample_in_subspace(n_directions, radius, random_seed=0):
    """
    Sample in the subspace spanned by the first n_directions PCA directions.
    """

    np.random.seed(random_seed)

    adjustment = np.zeros_like(n_directions[0])

    scale_factors = np.random.uniform(-1, 1, len(n_directions))
    scale_factors = (scale_factors / np.linalg.norm(scale_factors)) * radius

    # Create the adjustment vector by scaling and summing the PCA directions.
    for i, direction in enumerate(n_directions):
        adjustment += scale_factors[i] * direction

    return adjustment


def rescale_data(data):
    """
    Rescale the data to range between 0 and 1.
    :param data: The dataset to be rescaled.
    :return: Rescaled dataset.
    """
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)


def plot_combined_relationship(radii, data_series, ylabels, title, filename):
    """
    Plot multiple relationships on the same graph.

    :param radii: X-axis values.
    :param data_series: List of datasets to be plotted.
    :param ylabels: List of labels for each dataset.
    :param title: Title of the plot.
    :param filename: Path where the plot should be saved.
    """
    plt.figure(figsize=(12, 8))
    markers = ["o", "s", "^", "d"]

    for idx, data in enumerate(data_series):
        data = rescale_data(data)

        means = data.mean(axis=1)
        stds = data.std(axis=1)

        plt.plot(radii, means, marker=markers[idx], label=f"Mean of {ylabels[idx]}")
        plt.fill_between(radii, means - stds, means + stds, alpha=0.2)

    plt.xlabel("Radius")
    plt.ylabel("Value")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.xscale("log")  # Using a log scale for radii for better visualization
    # plt.yscale("log")
    plt.tight_layout()

    plt.savefig(filename, bbox_inches="tight")
    plt.close()
