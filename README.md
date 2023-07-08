# tiny-gen

This repository contains code for a Toy Model of generalisation to test claims regarding grokking, the over-parameterised regime, double descent etc.

## Tasks

We currently maintain one mathematical task to analyse generalisation. The so-called parity-prediction task.

> Given a sequence $x$ of length $n$ and a k_factor $k$, predict the parity of the number of 1s in the sequence $x[:k]$ given knowledge of the sequence $x[:k]$ and possibly the value of $k$.

## Experiments & Future Questions

Experiment descriptions and the asociated code can be seen in `tiny-gen.py`. Their results can be seen in the `experiments` folder.

A list of research questions:
* Can we show on multiple Grokking datasets that feature-learning is required for Grokking?
* Can we come up with an example where double grokking occurs? That is a model moves through an intermediate 'general' repsentation that's easier to learn but doesn't capture the full spectrum of the problem?
* How does Grokking interact with limiting neural network depth?
