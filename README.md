# tiny-gen

This repository contains code for a Toy Model of generalisation to test claims regarding grokking, the over-parameterised regime, double descent etc.

## Tasks

We currently maintain one mathematical task to analyse generalisation. The so-called parity-prediction task.

> Given a sequence $x$ of length $n$ and a k_factor $k$, predict the parity of the number of 1s in the sequence $x[:k]$ given knowledge of the sequence $x[:k]$ and possibly the value of $k$.

## Experiments & Future Questions

Experiment descriptions and the asociated code can be seen in `tiny-gen.py`. Their results can be seen in the `experiments` folder.

A list of research questions:
* Can we show on multiple Grokking datasets that feature-learning is required for Grokking?
* Can we come up with an example where double grokking occurs? That is a model moves through an intermediate 'general' repsentation that's easier to learn but doesn't capture the full spectrum of the problem? This is currently being done under `experiment_9`
* How does Grokking interact with limiting neural network depth?

## From Tegmark Paper...

Link: https://ericjmichaud.com/grokking-squared/

Quote: 

_There is a difficult coordination problem that the "early layers" and the "late layers" have to solve for this special internal operation to be learned. In particular, if the "late layers" learn much faster than the "early layers", then they will quickly fit bad, random, approximately static representations (given by the "early layers" at initialization), resulting in overfitting. On the other hand, if the "early layers" learn much faster than the "late layers", then they will quickly find (weird) representations which when thrown through the bad, random, approximately static "later layers" will produce the desired outputs (inverting the "later layers"). This will also result in overfitting._

_Generalization requires that the "early layers" and "late layers" coordinate well. In the case of grokking, there is a coordination failure at first. The network learns unstructured representations and fits them. But it can't do this perfectly (the training loss is not exactly zero), giving some training signal for the "early layers" and "late layers" to work with. We suggest that the "later layers" will be less complex, and able to achieve lower loss, if they learn to perform something akin to the underlying binary operation._

I think we can test this by freezing the starting layers at a certain interval, artifically rate limiting various components of the network.
