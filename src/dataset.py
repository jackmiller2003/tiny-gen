import torch
from torch.utils.data import Dataset
from typing import Tuple


class ParityPredictionDataset(Dataset):
    """
    Dataset for the parity prediction problem:

    Given a sequence of length n composed of -1 and 1,
    calculate the parity of the first k members of the sequence.

    For example, [1,1,-1,1,1,-1] with k=3 has a parity of -1
    since we only look at the first three components.
    We inform the network of the k and give it the sequence.
    """

    def __init__(
        self,
        num_samples: int,
        sequence_length: int,
        k_factor_range: tuple,
        max_k_factor: int = 20,
    ) -> None:
        """
        Initialises the dataset with:
        - num_samples: int
        - sequence_length: int
        - k_factor_range: tuple, the range of k factors to use before generalisation
        """

        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.k_factor_range = k_factor_range
        self.max_k_factor = max_k_factor

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.generate_dataset()

    def generate_dataset(self) -> None:
        """
        Generates the dataset by creating random sequences of -1 and 1
        then assigns to each sequence the parity of the first k members
        for k in the range k_factor_range.
        """

        # Create random sequences of -1 and 1
        sequences = (
            torch.randint(
                0, 2, size=(self.num_samples, self.sequence_length), device=self.device
            )
            * 2
            - 1
        )

        data_list = []
        label_list = []

        for sequence in sequences:
            for k_factor in range(self.k_factor_range[0], self.k_factor_range[1] + 1):
                # Calculate the parity of the first k_factor elements
                parity = torch.prod(sequence[:k_factor])

                # Create a one-hot encoding of k_factor
                k_factor_one_hot = torch.zeros(self.max_k_factor, device=self.device)
                k_factor_one_hot[k_factor - 1] = 1

                sequence_with_prepended_k = torch.cat((k_factor_one_hot, sequence))

                data_list.append(sequence_with_prepended_k)
                label_list.append(parity)

        self.data = torch.stack(data_list)
        self.labels = torch.stack(label_list)

    def __len__(self):
        return self.num_samples * (self.k_factor_range[1] - self.k_factor_range[0] + 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]
