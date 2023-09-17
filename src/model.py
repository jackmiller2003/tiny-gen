from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy.typing as npt
import numpy as np
import random
import os

os.sys.path.append("..")

from src.vbnn_layers import GaussianLinear


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
        normalise_output: bool = False,
        bias: bool = False,
        verbose: bool = True,
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
        self.normalise_output = normalise_output

        self.fc1 = nn.Linear(self.input_size, self.hidden_layer_size)
        self.fc2 = nn.Linear(self.hidden_layer_size, self.output_size, bias=bias)

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

    def forward(self, x, y=None):
        """
        Completes a forward pass of the network
        """

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        if self.normalise_output:
            x = F.log_softmax(x, dim=1)

        return x


class TinyBayes(nn.Module):
    """
    Small BNN for testing generalisation
    """

    def __init__(
        self,
        input_size: int,
        hidden_layer_size: int,
        output_size: int,
        random_seed: int,
        normalise_output: bool = False,
        verbose: bool = True,
        q_mean_std: float = 0.1,
    ):
        # Sets all random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        super(TinyBayes, self).__init__()

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.normalise_output = normalise_output

        fc1 = GaussianLinear([self.input_size, self.hidden_layer_size], q_mean_std=q_mean_std)
        fc2 = GaussianLinear([self.hidden_layer_size, self.output_size], q_mean_std=q_mean_std)
        self.layers = nn.ModuleList()
        self.layers.append(fc1)
        self.layers.append(fc2)

        # Determine if there is a GPU available and if so, move the model to GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        if verbose:
            print(f"Model initialised on device: {self.device}")

    def forward(self, x, y=None, local=True):
        """
        Completes a forward pass of the network
        """

        inputs = x
        for layer in self.layers:
            a = layer.forward(inputs, local)
            inputs = torch.relu(a)
        if self.normalise_output:
            a = F.log_softmax(a, dim=1)

        return a

    def kl(self):
        kl = 0
        for layer in self.layers:
            kl += layer.compute_variational_kl_term()
        return kl

    def look(self, layer: int) -> npt.NDArray[np.float64]:
        """
        Looks inside the model weights producing
        """

        if layer == 1:
            return self.layers[0].weight_sample()[0].cpu().detach().numpy()
        elif layer == 2:
            return self.layers[1].weight_sample()[0].cpu().detach().numpy()
        else:
            raise ValueError("Invalid layer.")


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
