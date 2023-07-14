import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from typing import Tuple
import random
import numpy as np
from tqdm import tqdm
from scipy.integrate import odeint


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

        self._generate_data()

    def _generate_data(self) -> None:
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

        self._generate_data()

    def _generate_data(self) -> None:
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

        self._generate_data()

    def _generate_data(self) -> None:
        """
        Generates the data for the modulo addition task.

        The input data should be a sequence of lenght 2*modulo
        representing a and b in the equation
        a+b = c (mod modulo)

        Here a and b are one hot encoded.
        """

        self.data = torch.zeros((self.num_samples, 2 * self.modulo))
        self.targets = torch.zeros((self.num_samples, self.modulo))

        for i in range(0, self.num_samples):
            a = random.randint(0, self.modulo - 1)
            b = random.randint(0, self.modulo - 1)
            c = (a + b) % self.modulo

            a_zeros = torch.zeros(self.modulo)
            b_zeros = torch.zeros(self.modulo)
            c_zeros = torch.zeros(self.modulo)

            a_zeros[a] = 1
            b_zeros[b] = 1
            c_zeros[c] = 1

            self.data[i] = torch.cat((a_zeros, b_zeros))
            self.targets[i] = c_zeros

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
        assert self.total_length >= len(dataset[0][0])

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


def combine_datasets(
    dataset1: Dataset, dataset2: Dataset, individual: bool = True
) -> Dataset:
    """
    Combine two pytorch datasets such that the combined dataset has an
    input space of input(dataset1) + input(dataset2) and an output
    space of output(dataset1) + output(dataset2)

    From GPT-4...
    """
    combined_data = []
    combined_labels = []

    if individual:
        # Iterate over first dataset and add examples. Just pad with zeros
        # for the second dataset.

        total_input_zeros = len(dataset2[0][0]) + len(dataset1[0][0])
        total_label_zeros = len(dataset2[0][1]) + len(dataset1[0][1])

        for input, target in dataset1:
            zeros = torch.zeros(total_input_zeros)
            # print(f"input.shape: {input.shape}")
            # print(f"zeros.shape: {zeros.shape}")
            # print(zeros[: input.shape[0]])
            zeros[: input.shape[0]] += input
            combined_data.append(zeros)

            zeros = torch.zeros(total_label_zeros)
            zeros[: target.shape[0]] += target
            combined_labels.append(zeros)

        for inputs, targets in dataset2:
            zeros = torch.zeros(total_input_zeros)
            zeros[dataset1[0][0].shape[0] :] = inputs
            combined_data.append(zeros)

            zeros = torch.zeros(total_label_zeros)
            zeros[dataset1[0][1].shape[0] :] = targets
            combined_labels.append(zeros)

        combined_data = torch.stack(combined_data)
        combined_labels = torch.stack(combined_labels)

    else:
        assert len(dataset1) == len(dataset2)

        for i in range(len(dataset1)):
            data1, labels1 = dataset1[i]
            data2, labels2 = dataset2[i]

            combined_data.append(torch.cat((data1, data2)))
            combined_labels.append(torch.cat((labels1, labels2)))

        combined_data = torch.stack(combined_data)
        combined_labels = torch.stack(combined_labels)

    return TensorDataset(combined_data, combined_labels)


class PendulumPredictionTask(Dataset):
    """
    The Pendulum prediction task generates initial conditions for a pendulum
    and integrates the equations of motion to generate a trajectory for each initial condition.
    Each sample in the dataset corresponds to a single time step.
    """

    def __init__(
        self,
        gravity: float,
        length: float,
        num_samples: int,
        time_step: float,
        random_seed: int,
        progress_bar: bool = True,
    ) -> None:
        self.gravity = gravity
        self.length = length
        self.num_samples = num_samples
        self.time_step = time_step
        self.progress_bar = progress_bar

        # Setting random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        self._generate_data()

    def _simple_pendulum_deriv(self, x, t):
        # The simple pendulum subject to zero damping and zero control input
        nx = np.zeros(2)
        nx[0] = x[1]
        nx[1] = -(self.gravity / self.length) * np.sin(x[0])
        return nx

    def _generate_data(self):
        t_span = np.linspace(0, self.time_step, 2)
        self.data = torch.zeros((self.num_samples, 2))
        self.targets = torch.zeros((self.num_samples, 2))

        for i in tqdm(range(self.num_samples), disable=not self.progress_bar):
            initial_angle = np.random.uniform(-np.pi, np.pi)
            initial_velocity = np.random.uniform(-1, 1)
            sol = odeint(
                self._simple_pendulum_deriv,
                y0=[initial_angle, initial_velocity],
                t=t_span,
            )

            self.data[i] = torch.tensor(sol[0])
            self.targets[i] = torch.tensor(sol[1])

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]

    def __name__(self) -> str:
        return "PendulumPredictionTask"


if __name__ == "__main__":
    pendulum_dataset = PendulumPredictionTask(
        gravity=9.8,
        length=1,
        num_samples=1000,
        time_step=0.01,
        random_seed=42,
    )

    print(pendulum_dataset[0])

    # Graph the pendulum dataset
    import matplotlib.pyplot as plt

    plt.plot(pendulum_dataset.data[:, 0], pendulum_dataset.data[:, 1], "o")

    plt.savefig("pendulum.png")
