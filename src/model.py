import torch
import torch.nn as nn

import numpy.typing as npt
import numpy as np


class TinyModel(nn.Module):
    """
    Small model for testing generalisation.
    """

    def __init__(
        self,
        input_size: int,
        hidden_layer_size: int,
        output_size: int,
        random_seed: int = 42,
        verbose: bool = True,
    ) -> None:
        """
        Initialises network with parameters:
        - input_size: int
        - output_size: int
        - hidden layer: int
        """

        super(TinyModel, self).__init__()

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_layer_size)
        self.fc2 = nn.Linear(self.hidden_layer_size, self.output_size, bias=False)

        torch.manual_seed(random_seed)

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


class BigModel(nn.Module):
    """
    Model with more than two layers :O
    """

    def __init__(
        self,
        input_size: int,
        hidden_layer_sizes: list[int],
        output_size: int,
        random_seed: int = 42,
        verbose: bool = True,
    ) -> None:
        super(BigModel, self).__init__()

        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.random_seed = random_seed

        self.fc1 = nn.Linear(self.input_size, self.hidden_layer_sizes[0])
        self.fc2 = nn.Linear(
            self.hidden_layer_sizes[0], self.hidden_layer_sizes[1], bias=True
        )
        self.fc3 = nn.Linear(self.hidden_layer_sizes[1], self.output_size, bias=False)

        torch.manual_seed(random_seed)

        # Initialise weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

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
        elif layer == 3:
            return self.fc3.weight.cpu().detach().numpy()
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
                self.fc2.bias.requires_grad = False
            elif layer == 3:
                self.fc3.weight.requires_grad = False
                self.fc3.bias.requires_grad = False
            else:
                raise ValueError("Invalid layer.")

    def forward(self, x):
        """
        Completes a forward pass of the network
        """

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

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

        self.layers = nn.ModuleList()

        previous_layer_size = self.input_size
        for layer_size in self.hidden_layer_sizes:
            self.layers.append(nn.Linear(previous_layer_size, layer_size))
            previous_layer_size = layer_size

        self.layers.append(nn.Linear(previous_layer_size, self.output_size))

        torch.manual_seed(random_seed)

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
