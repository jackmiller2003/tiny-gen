import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from typing import Tuple
import random
import numpy as np


class ParityTask(Dataset):
    """
    The parity prediction task takes in binary sequences and provides
    the parity of the sequence as the target.

    For example:

    [0, 1, 0, 1] -> [1,0]
    [1, 1, 0, 1] -> [0,1]
    """

    def __init__(
        self, sequence_length: int, num_samples: int, random_seed: int
    ) -> None:
        self.sequence_length = sequence_length
        self.num_samples = num_samples
        self.random_seed = random_seed

        # Setting random seeds
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        self.generate_data()

    def generate_data(self) -> None:
        """
        Generates the data for the parity prediction task.

        Here the result should be one hot encoded into two classes,
        even and odd.
        """
        self.data = torch.randint(
            0,
            2,
            (self.num_samples, self.sequence_length),
        )

        parities = []

        for data_point_sequence in self.data:
            new_sequence = []
            for point in data_point_sequence:
                if point == 0:
                    new_sequence.append(-1)
                else:
                    new_sequence.append(1)

            one_hot_parity = torch.zeros(2)

            parity = torch.prod(torch.tensor(new_sequence))

            if parity == -1:
                one_hot_parity[0] = 1
            else:
                one_hot_parity[1] = 1

            parities.append(one_hot_parity)

        self.targets = torch.stack(parities)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


class PeekParityTask(Dataset):
    """
    This is similar to the parity prediction task. However, upon a certain set of
    set of conditions, the network must peek at the (k+1)th value to determine
    the parity.
    """

    def __init__(
        self,
        sequence_length: int,
        num_samples: int,
        peek_condition: list[int],
        random_seed: int,
    ) -> None:
        """
        Note sequence lenght is the usual lenght of the sequence, not the
        peek ahead sequence length.
        """
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.random_seed = random_seed
        self.peek_condition = peek_condition

        # Setting random seeds
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        if len(peek_condition) != sequence_length:
            raise ValueError("Peek condition must be the same length as k.")

        self.generate_data()

    def generate_data(self) -> None:
        self.data = torch.randint(
            0,
            2,
            (self.num_samples, self.sequence_length + 1),
        )

        parities = []

        for data_point_sequence in self.data:
            new_sequence = []
            for point in data_point_sequence[:-1]:
                if point == 0:
                    new_sequence.append(-1)
                else:
                    new_sequence.append(1)

            one_hot_parity = torch.zeros(2)

            if new_sequence == self.peek_condition:
                new_sequence.append(data_point_sequence[-1])

            parity = torch.prod(torch.tensor(new_sequence))

            if parity == -1:
                one_hot_parity[0] = 1
            else:
                one_hot_parity[1] = 1

            parities.append(one_hot_parity)

        self.targets = torch.stack(parities)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


class ModuloAdditionTask(Dataset):
    """
    The modulo addition task takes in binary sequences and provides
    the modulo addition of the sequence as the target.
    """

    def __init__(self, modulo: int, num_samples: int, random_seed: int) -> None:
        self.modulo = modulo
        self.num_samples = num_samples
        self.random_seed = random_seed

        # Setting random seeds
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        self.generate_data()

    def generate_data(self) -> None:
        """
        Generates the data for the modulo addition task.

        The input data should be a sequence of lenght 2*modulo
        representing a and b in the equation
        a+b = c (mod modulo)

        Here a and b are one hot encoded.
        """

        for i in range(0, self.num_samples):
            a = random.randint(0, self.modulo - 1)
            b = random.randint(0, self.modulo - 1)
            c = (a + b) % self.modulo
            self.data[i] = torch.cat(
                (
                    torch.nn.functional.one_hot(a, self.modulo),
                    torch.nn.functional.one_hot(b, self.modulo),
                )
            )
            self.targets[i] = torch.nn.functional.one_hot(c, self.modulo)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


class HiddenDataset(Dataset):
    """
    Hides a dataset by increasing the sequence length with random
    0s and 1s at the end of the sequence. The total length of the
    sequence is given by the hidden_length parameter.
    """

    def __init__(self, dataset: Dataset, total_length: int, random_seed: int) -> None:
        self.dataset = dataset
        self.total_length = total_length
        self.random_seed = random_seed

        # Setting random seeds
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # Esnure the total length is greater than the sequence length
        assert self.total_length > len(dataset[0][0])

        self.generate_new_examples()

    def generate_new_examples(self) -> None:
        new_examples = []
        new_targets = []

        for data_point, target in self.dataset:
            new_data_point = torch.zeros(self.total_length)
            new_data_point[: len(data_point)] = data_point[: len(data_point)]
            new_data_point[len(data_point) :] = torch.randint(
                0, 2, (self.total_length - len(data_point), 1)
            ).squeeze()
            new_examples.append(new_data_point)
            new_targets.append(target)

        self.data = torch.stack(new_examples)
        self.targets = torch.stack(new_targets)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]
