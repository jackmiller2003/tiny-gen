## Old Experiments

This file contains a list of old experiment descriptions that have been previously run but don't currently exist within `tiny-gen.py`.

### 1. Multiple k values

I want to see at what point the network is able to generalise beyond
the training data to complete the task of parity prediction.

To do this, we start with 3 random seeds and walk through the k ranges:
* [2,2]
* [2,3]
* [2,4]
* [2,5]

Then, we look at the test results on the generalisation dataset which goes from
k=6 to k=10. We will use a sequence length of 10.

### 2. Grokking with multiple k values

The aim of this experiment is to determine whether we get grokking behaviour.

That is, train on k=2,3,4,5 and then at each epoch test on k=6. If we were to see
grokking then we would expect to see the accuracy on k=6 to increase drastically
at one point.

### 3. Grokking with transplant-initialisation

This experiment involves `transplate-initialisation`. What happens when take a learned dense subnetwork and transplant partials components of it into a new network? Does Grokking still occur? Likely not.