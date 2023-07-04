import torch
import torch.nn as nn


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

    def forward(self, x):
        """
        Completes a forward pass of the newtork
        """

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# Taken from https://arxiv.org/pdf/2303.11873.pdf
class MyHingeLoss(torch.nn.Module):
    def __init__(self):
        super(MyHingeLoss, self).__init__()

    def forward(self, output, target):
        # print(f"output: {torch.squeeze(output)}")
        # print(f"target: {torch.squeeze(target)}")
        multiplied_vector = torch.mul(torch.squeeze(output), torch.squeeze(target))
        # print(f"multiplied_vector: {multiplied_vector}")

        hinge_loss = 1 - multiplied_vector
        hinge_loss[hinge_loss < 0] = 0

        # print(f"hinge_loss: {hinge_loss}")

        return hinge_loss.mean()
