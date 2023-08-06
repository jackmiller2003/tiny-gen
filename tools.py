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
        init_ll = [-6, -6, 3]
        init_ls = [-4, 4, 0]
        colors = ["g", "y", "b"]
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

                if i in epochs_ls:
                    lengthscales.append(
                        model.covar_module.base_kernel.lengthscale.detach()
                        .cpu()
                        .numpy()
                    )
                    model.eval()
                    x = torch.from_numpy(np.linspace(-2, 2, 100)).to(device)
                    plt.figure(0)
                    plt.plot(train_inputs.cpu(), train_targets.cpu(), "+k")
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

                    plt.savefig(path_to_plot / Path(f"gpr_{init_idx}_pred_{i}.pdf"))

                optimiser.step()

            # --- Adding trajectories to existing landscape --- #

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

            plt.figure(0)
            plt.plot(np.arange(epochs) + 1, train_mses, "-r", label="train")
            plt.plot(np.arange(epochs) + 1, valid_mses, "-b", label="validation")
            plt.xscale("log")
            plt.xlabel("epoch")
            plt.ylabel("mse")
            plt.legend()
            plt.savefig(path_to_plot / Path(f"gpr_{init_idx}_mse.pdf"))

            plt.figure(0)
            plt.plot(np.arange(epochs) + 1, train_lps, "-r", label="train")
            plt.plot(np.arange(epochs) + 1, valid_lps, "-b", label="validation")
            plt.xscale("log")
            plt.xlabel("epoch")
            plt.ylabel("log prob")
            plt.legend()
            plt.savefig(path_to_plot / Path(f"gpr_{init_idx}_lp.pdf"))

            plt.figure(0)
            plt.plot(np.arange(epochs) + 1, lmls, "-k")
            plt.plot(np.arange(epochs) + 1, fits, "-b")
            plt.plot(np.arange(epochs) + 1, comps, "-r")
            plt.xscale("log")
            plt.xlabel("epoch")
            plt.ylabel("objective, data fit and complexity")
            plt.savefig(path_to_plot / Path(f"gpr_{init_idx}_lml.pdf"))
