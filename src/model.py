from __future__ import annotations

import torch
import torch.nn as nn
import gpytorch
from gpytorch.constraints import Positive

import numpy.typing as npt
import numpy as np
import random
import math

torch.manual_seed(42)
np.random.seed(42)

import gpytorch
import matplotlib.pyplot as plt
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    UnwhitenedVariationalStrategy,
)
from tqdm import tqdm


class TinyModel(nn.Module):
    """
    Small model for testing generalisation.
    """

    def __init__(
        self,
        input_size: int,
        hidden_layer_size: int,
        output_size: int,
        random_seed: int,
        verbose: bool = False,
    ) -> None:
        """
        Initialises network with parameters:
        - input_size: int
        - output_size: int
        - hidden layer: int
        """

        # Sets all random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        super(TinyModel, self).__init__()

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_layer_size)
        self.fc2 = nn.Linear(self.hidden_layer_size, self.output_size, bias=False)

        # Initialise weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

        # Determine if there is a GPU available and if so, move the model to GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        if verbose:
            print(f"Model initialised on device: {self.device}")

    def look(self, layer: int) -> npt.NDArray[np.float64]:
        """
        Looks inside the model weights producing
        """

        if layer == 1:
            return self.fc1.weight.cpu().detach().numpy()
        elif layer == 2:
            return self.fc2.weight.cpu().detach().numpy()
        else:
            raise ValueError("Invalid layer.")

    def freeze(self, list_of_layers: list[int]) -> None:
        """
        Freezes layers in the model.
        """

        for layer in list_of_layers:
            if layer == 1:
                self.fc1.weight.requires_grad = False
                self.fc1.bias.requires_grad = False
            elif layer == 2:
                self.fc2.weight.requires_grad = False
            else:
                raise ValueError("Invalid layer.")

    def unfreeze(self, list_of_layers: list[int]) -> None:
        """
        Unfreezes layers in the model.
        """

        for layer in list_of_layers:
            if layer == 1:
                self.fc1.weight.requires_grad = True
                self.fc1.bias.requires_grad = True
            elif layer == 2:
                self.fc2.weight.requires_grad = True
            else:
                raise ValueError("Invalid layer.")

    def forward(self, x):
        """
        Completes a forward pass of the network
        """

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class ExpandableModel(nn.Module):
    """
    Model with an expandable number of layers
    """

    def __init__(
        self,
        input_size: int,
        hidden_layer_sizes: list[int],
        output_size: int,
        random_seed: int = 42,
        verbose: bool = True,
    ) -> None:
        super(ExpandableModel, self).__init__()

        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.random_seed = random_seed

        # Sets all random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        self.layers = nn.ModuleList()

        previous_layer_size = self.input_size
        for layer_size in self.hidden_layer_sizes:
            self.layers.append(nn.Linear(previous_layer_size, layer_size))
            previous_layer_size = layer_size

        self.layers.append(nn.Linear(previous_layer_size, self.output_size, bias=False))

        # Initialise weights
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)

        # Determine if there is a GPU available and if so, move the model to GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.to(self.device)

        if verbose:
            print(f"Model initialised on device: {self.device}")

    def look(self, layer: int) -> npt.NDArray[np.float64]:
        """
        Looks inside the model weights producing
        """

        if layer <= len(self.layers):
            return self.layers[layer - 1].weight.cpu().detach().numpy()
        else:
            raise ValueError("Invalid layer.")

    def freeze(self, list_of_layers: list[int]) -> None:
        """
        Freezes layers in the model.
        """

        for layer in list_of_layers:
            if layer <= len(self.layers):
                self.layers[layer - 1].weight.requires_grad = False
                self.layers[layer - 1].bias.requires_grad = False
            else:
                raise ValueError("Invalid layer.")

    def forward(self, x):
        """
        Completes a forward pass of the network
        """

        for i in range(len(self.layers) - 1):
            x = torch.relu(self.layers[i](x))

        x = self.layers[-1](x)

        return x


class TinyLinearModel(nn.Module):
    """
    Small model for testing generalisation.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        random_seed: int,
        verbose: bool = False,
    ) -> None:
        """
        Initialises network with parameters:
        - input_size: int
        - output_size: int
        - hidden layer: int
        """

        # Sets all random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        super(TinyLinearModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, self.output_size, bias=False)

        # Initialise weights
        nn.init.xavier_uniform_(self.fc1.weight)

        # Determine if there is a GPU available and if so, move the model to GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        if verbose:
            print(f"Model initialised on device: {self.device}")

    def forward(self, x):
        """
        Completes a forward pass of the network
        """

        x = self.fc1(x)

        return x


class RBFLinearModel(torch.nn.Module):
    def __init__(self, rbf_means: list, rbf_variance=0.2):
        """
        Initialises a linear model with radial basis functions (with help from Chat-GPT)
        """
        super(RBFLinearModel, self).__init__()

        self.rbf_means = torch.tensor(rbf_means).float()
        self.rbf_variance = rbf_variance

        self.weights = torch.nn.Parameter(torch.randn(len(rbf_means), 1))
        self.bias = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        """
        Completes a forward pass of the network
        """
        # Create radial basis functions
        rbf_functions = []
        for mean in self.rbf_means:
            rbf_functions.append(
                torch.exp(-torch.pow(x - mean, 2) / (2 * self.rbf_variance))
            )

        # Concatenate radial basis functions
        x_rbf = torch.concat(rbf_functions, dim=1)

        # Linear combination of the RBFs
        out = self.weights.T @ x_rbf.T

        return out


# Taken from https://arxiv.org/pdf/2303.11873.pdf
class MyHingeLoss(torch.nn.Module):
    def __init__(self):
        super(MyHingeLoss, self).__init__()

    def forward(self, output, target):
        # Ensure that the output and target tensors are on the same device
        output = output.to(target.device)

        multiplied_vector = torch.mul(torch.squeeze(output), torch.squeeze(target))

        hinge_loss = 1 - multiplied_vector
        hinge_loss[hinge_loss < 0] = 0

        return hinge_loss.mean()


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_inputs: torch.Tensor,
        train_targets: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        num_dimensions: int,
    ):
        super(ExactGPModel, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=num_dimensions,
                lengthscale_constraint=Positive(torch.exp, torch.log),
            ),
            outputscale_constraint=Positive(torch.exp, torch.log),
        )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ApproxGPModel(ApproximateGP):
    def __init__(self, x_train):
        N, D = x_train.size(0), x_train.size(1)
        var_dist = CholeskyVariationalDistribution(N)
        var_stra = UnwhitenedVariationalStrategy(
            self, x_train, var_dist, learn_inducing_locations=False
        )
        super(ApproxGPModel, self).__init__(var_stra)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=D)
        )

    def forward(self, x):
        x_mean = self.mean_module(x)
        x_covar = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(x_mean, x_covar)
        return latent_pred


def mvn_log_prob(dist, value: torch.Tensor) -> torch.Tensor:
    r"""
    See :py:meth:`torch.distributions.Distribution.log_prob
    <torch.distributions.distribution.Distribution.log_prob>`.
    """
    if gpytorch.settings.fast_computations.log_prob.off():
        return super(type(dist), dist).log_prob(value)

    if dist._validate_args:
        dist._validate_sample(value)

    mean, covar = dist.loc, dist.lazy_covariance_matrix
    diff = value - mean

    # Repeat the covar to match the batch shape of diff
    if diff.shape[:-1] != covar.batch_shape:
        if len(diff.shape[:-1]) < len(covar.batch_shape):
            diff = diff.expand(covar.shape[:-1])
        else:
            padded_batch_shape = (
                *(1 for _ in range(diff.dim() + 1 - covar.dim())),
                *covar.batch_shape,
            )
            covar = covar.repeat(
                *(
                    diff_size // covar_size
                    for diff_size, covar_size in zip(
                        diff.shape[:-1], padded_batch_shape
                    )
                ),
                1,
                1,
            )

    # Get log determininant and first part of quadratic form
    covar = covar.evaluate_kernel()
    inv_quad, logdet = covar.inv_quad_logdet(
        inv_quad_rhs=diff.unsqueeze(-1), logdet=True
    )

    res = -0.5 * sum([inv_quad, logdet, diff.size(-1) * math.log(2 * math.pi)])
    return res, -0.5 * inv_quad, -0.5 * logdet


class ExactMarginalLikelihood(gpytorch.mlls.ExactMarginalLogLikelihood):
    def forward(self, function_dist, target, *params):
        r"""
        Computes the MLL given :math:`p(\mathbf f)` and :math:`\mathbf y`.

        :param ~gpytorch.distributions.MultivariateNormal function_dist: :math:`p(\mathbf f)`
            the outputs of the latent function (the :obj:`gpytorch.models.ExactGP`)
        :param torch.Tensor target: :math:`\mathbf y` The target values
        :rtype: torch.Tensor
        :return: Exact MLL. Output shape corresponds to batch shape of the model/input data.
        """
        if not isinstance(function_dist, gpytorch.distributions.MultivariateNormal):
            raise RuntimeError(
                "ExactMarginalLogLikelihood can only operate on Gaussian random variables"
            )

        # Get the log prob of the marginal distribution
        output = self.likelihood(function_dist, *params)
        res, fit_term, complexity_term = mvn_log_prob(output, target)
        res = self._add_other_terms(res, params)

        # Scale by the amount of data we have
        num_data = function_dist.event_shape.numel()
        return (
            res.div_(num_data),
            fit_term.div_(num_data),
            complexity_term.div_(num_data),
        )
