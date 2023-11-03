from parallel_trunc_gaussian import ParallelTruncatedGaussian as TG
import torch
from torch import tensor as t
from torch.distributions.kl import kl_divergence
from scipy.stats import truncnorm, entropy
import numpy as np

def compare_var(mu, sigma, a, b):
    v1 = TG(t(mu), t(sigma), t(a), t(b)).variance

    a_, b_ = (a - mu) / sigma, (b - mu) / sigma
    v2 = truncnorm.var(a=a_, b=b_, loc=mu, scale=sigma)

    return v2, v1

def scipy_kl(d1_params, d2_params):
    mu1, sigma1, a1, b1 = d1_params
    mu2, sigma2, a2, b2 = d2_params

    # Normalize parameters
    a1_, b1_ = (a1 - mu1) / sigma1, (b1 - mu1) / sigma1
    a2_, b2_ = (a2 - mu2) / sigma2, (b2 - mu2) / sigma2

    d1_scipy = truncnorm(a=a1_, b=b1_, loc=mu1, scale=sigma1)
    d2_scipy = truncnorm(a=a2_, b=b2_, loc=mu2, scale=sigma2)

    n_points = int(1e5)
    x = np.linspace(d2_scipy.ppf(1e-5), d2_scipy.ppf(1-1e-5), n_points)

    pdf1 = d1_scipy.pdf(x)
    pdf2 = d2_scipy.pdf(x)

    return entropy(pdf1, pdf2)

def compare_kl(d1_params, d2_params):
    mu1, sigma1, a1, b1 = d1_params
    mu2, sigma2, a2, b2 = d2_params

    d1 = TG(t(mu1), t(sigma1), t(a1), t(b1))
    d2 = TG(t(mu2), t(sigma2), t(a2), t(b2))

    kl = kl_divergence(d1, d2)
    kl_scipy = scipy_kl(d1_params, d2_params)

    return kl_scipy, kl

def compare_kl_and_var(params_list):
    for i, params in enumerate(params_list):
        kl_scipy, kl = compare_kl(params[0], params[1])
        v_scipy1, v1 = compare_var(params[0][0], params[0][1], params[0][2], params[0][3])
        v_scipy2, v2 = compare_var(params[1][0], params[1][1], params[1][2], params[1][3])

        print(f"\n------ Number {i} ------")
        print(f"Params: {params}")
        print(f"KL divergence: {kl_scipy} vs {kl}")
        print(f"Var 1: {v_scipy1} vs {v1}")
        print(f"Var 2: {v_scipy2} vs {v2}")
        print("------------------")

# Example params to test kl and var
# In order: first distribution, second distribution
# Each distribution has mu,sigma,a,b
# Important Note: the [a,b] interval of the first distribution must be a subset of the second distribution (a2 <= a1 <= b1 <= b2)

# Generate params where both distributions always have mu=0 and sigma=1
"""params_list=[
    ((30.0764, 1.4142, 29.9000, 100000.1016), (30.0764, 1.4142, 29.9000, 100000.1016)),
    ((30.0764, 1.4142, 29.9000, 100000.1016), (31, 1.1000, 29.9000, 100000.1016)),
    ((30.0764, 1.4142, 29.9000, 100000.1016), (30.05, 1.4, 29.9000, 100000.1016)),
    ((3,1,-5,10000), (4,1.5,-5,10000)),
    ((4,1.5,-5,10000), (3,1,-5,10000)),

    ((0, 1, -1, 1), (0, 1, -1, 1)),
    ((0, 1, -1, 1), (0, 1, -10, 10)),
    ((0, 1, -1, 1), (0, 1, -1, 10)),
    ((0, 1, -1, 1), (0, 1, -10, 1)),
    ((0, 1, -10, 1), (0, 1, -10, 10)),
    ((0, 1, -100, 0), (0, 1, -101, 32.45)),

    ((0, 1, -1, 1), (0, 1, -1, 1e5)),
    ((0, 1, -1, 1), (0, 1, -1e5, 1)),
    ((0, 1, -1, 1), (0, 1, -1e6, 1e6)),

    ((0, 1, 10, 20), (0, 1, 10, 20)),
    ((0, 1, 10, 20), (0, 1, 10, 100)),
    ((0, 1, 10, 20), (0, 1, -10, 20)),
    ((0, 1, 10, 20), (0, 1, -10, 100)),
]"""

params_list=[
    ((0, 1, -1, 1), (0, 1, -1, 1)),
    ((0, 1, -1, 1), (0, 1, -10, 10)),
    ((0, 1, -1, 1), (0, 1, -1, 10)),
    ((0, 1, -1, 1), (0, 1, -10, 1)),

    ((0.5,2,-1,1), (0.5,2,-1,1)),
    ((0.5,2,-1,1), (0.5,2,-5,1)),
    ((0.5,2,-1,1), (0.5,2,-1,5)),

    ((1,2,-10,10), (0.5,2,-10,10)),
    ((2,4,-10,10), (0.5,4,-10,10)),
    ((3,8,-10,10), (0.5,8,-10,10)),
]

if __name__ == "__main__":
    compare_kl_and_var(params_list)


# TODO
"""
- TG and scipy return almost the same variance
- KL calculations are correct when mu=0, sigma=1
- Normalizing a,b as in scipy before calculating KL with truncated does not work


"""