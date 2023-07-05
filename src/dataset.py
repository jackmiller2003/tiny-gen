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
        verbose: bool = True,
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
        self.num_k_factors = self.k_factor_range[1] - self.k_factor_range[0] + 1

        self.generate_dataset()

    def generate_dataset(self) -> None:
        """
        Generates the dataset by creating random sequences of -1 and 1
        then assigns to each sequence the parity of the first k members
        for k in the range k_factor_range.
        """

        # Adjust the number of sequences based on the number of k_factors
        num_sequences = self.num_samples // self.num_k_factors

        sequences = (
            torch.randint(
                0, 2, size=(num_sequences, self.sequence_length)
            )
            * 2
            - 1
        )

        data_list = []
        label_list = []

        k_factors = list(range(self.k_factor_range[0], self.k_factor_range[1] + 1))

        for sequence in sequences:
            for k_factor in k_factors:
                # Calculate the parity of the first k_factor elements
                parity = torch.prod(sequence[:k_factor])

                # Create a one-hot encoding of k_factor
                k_factor_one_hot = torch.zeros(self.max_k_factor)
                k_factor_one_hot[k_factor - 1] = 1

                sequence_with_prepended_k = torch.cat((k_factor_one_hot, sequence))

                data_list.append(sequence_with_prepended_k)
                label_list.append(parity)

                # Once we reach the desired number of samples, stop adding more
                if len(data_list) >= self.num_samples:
                    break

            if len(data_list) >= self.num_samples:
                break

        self.data = torch.stack(data_list)[:self.num_samples]
        self.labels = torch.stack(label_list)[:self.num_samples]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]
    
class HiddenParityPrediction(Dataset):
    """
    In this parity prediction task the value of k is not given to the network.
    Instead the network must infer this on its own.
    """

    def __init__(
        self,
        num_samples: int,
        sequence_length: int,
        k: int,
    ):
        
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.k = k

        self.generate_dataset()

    def generate_dataset(self) -> None:
        """
        Generates the dataset by creating random sequences of -1 and 1
        then assigns to each sequence the parity of the first k members
        """

        # Create random sequences of -1 and 1
        sequences = (
            torch.randint(
                0, 2, size=(self.num_samples, self.sequence_length)
            )
            * 2
            - 1
        ).float()

        data_list = []
        label_list = []

        for sequence in sequences:
            # Calculate the parity of the first k_factor elements
            parity = torch.prod(sequence[:self.k])

            data_list.append(sequence)
            label_list.append(parity)

        self.data = torch.stack(data_list)
        self.labels = torch.stack(label_list)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]
