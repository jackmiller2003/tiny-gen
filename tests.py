import torch
from src.dataset import ParityPredictionDataset


def tests_parity_generation():
    """
    Tests parity generation.

    TODO: There's probably some way to do better testing.
    """

    number_tests = 1
    tests_passed = 0

    # Test 1
    num_samples = 1000
    sequence_length = 3
    k_factor_range = (1, 3)

    dataset = ParityPredictionDataset(
        num_samples=num_samples,
        sequence_length=sequence_length,
        k_factor_range=k_factor_range,
    )

    # k = 1 and sequence = [1, 1, -1] should have parity 1
    target_sequence = torch.tensor([1, 0, 0, 1, 1, -1])
    target_parity = torch.tensor(1)

    # Looks for a sequence which matches the target sequence and then
    # checks that the parity and k are correct.

    passed_test_1 = True

    for i in range(num_samples):
        if torch.equal(dataset[i][0][:3], target_sequence):
            passed_test_1 = passed_test_1 and torch.equal(dataset[i][1], target_parity)

    if passed_test_1:
        tests_passed += 1

    return tests_passed, number_tests


if __name__ == "__main__":
    """
    This is intended to complete some small tests on dataset generation.
    """

    tests_passed, number_tests = tests_parity_generation()

    print(f"Passed {tests_passed}/{number_tests} tests.")
