`[![PyPI version](https://badge.fury.io/py/stable-trunc-gaussian.svg)](https://badge.fury.io/py/stable-trunc-gaussian)
[![Downloads](https://static.pepy.tech/badge/stable-trunc-gaussian)](https://pepy.tech/project/stable-trunc-gaussian)

# Stable Truncated Gaussian
A **differentiable** implementation of the **Truncated Gaussian (Normal)** distribution using Python and Pytorch, which is **numerically stable** even when the *μ* parameter lies outside the interval *\[a,b\]* given by the bounds of the distribution. In this situation, a naive evaluation of the mean, variance and log-probability of the distribution could otherwise result in [catastrophic cancellation](https://en.wikipedia.org/wiki/Catastrophic_cancellation). Our code is inspired by [TruncatedNormal.jl](https://github.com/cossio/TruncatedNormal.jl) and [torch_truncnorm](https://github.com/toshas/torch_truncnorm). Currently, we only provide functionality for calculating the mean, variance and log-probability, but not for calculating the entropy or sampling from the distribution.

## Installation

Simply install with `pip`:

    pip install stable-trunc-gaussian

## Example

Run the following code in Python:

    from stable_trunc_gaussian import TruncatedGaussian as TG
    from torch import tensor as t
    
    # Create a Truncated Gaussian with mu=0, sigma=1, a=10, b=11
    # Notice how mu is outside the interval [a,b]
    dist = TG(t(0),t(1),t(10),t(11))
    
    print("Mean:", dist.mean)
    print("Variance:", dist.variance)
    print("Log-prob(10.5):", dist.log_prob(t(10.5)))
    
Result:

    Mean: tensor(10.0981)
    Variance: tensor(0.0094)
    Log-prob(10.5): tensor(-2.8126)

## Parallel vs Sequential Implementation
The class obtained by doing `from stable_trunc_gaussian import TruncatedGaussian` corresponds to a **parallel** implementation of the truncated gaussian, which makes possible to obtain several values (mean, variance and log-probs) in parallel. In case you are only interested in computing values sequentially, i.e., one at a time, we also provide a **sequential** implementation which results more efficient *only* for this case. In order to use this sequential implementation, simply do `from stable_trunc_gaussian import SeqTruncatedGaussian`. Here is an example:

    from stable_trunc_gaussian import TruncatedGaussian, SeqTruncatedGaussian
    from torch import tensor as t

    # Parallel computation
    means = TruncatedGaussian(t([0,0.5]),t(1,1),t(-1,2),t(1,5)).mean

    # Sequential computation
    # Note: the 'TruncatedGaussian' class can also be used for this sequential case
    mean_0 = SeqTruncatedGaussian(t([0]),t(1),t(-1),t(1)).mean
    mean_1 = SeqTruncatedGaussian(t([0.5]),t(1),t(2),t(5)).mean
