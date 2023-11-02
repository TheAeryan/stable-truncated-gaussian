from parallel_trunc_gaussian import ParallelTruncatedGaussian as TG
import torch
from torch import tensor as t
from torch.distributions.kl import kl_divergence
from scipy.stats import truncnorm

def calculate_var(mu, sigma, a, b):
    v1 = TG(t(mu), t(sigma), t(a), t(b)).variance

    a_, b_ = (a - mu) / sigma, (b - mu) / sigma
    v2 = truncnorm.var(a=a_, b=b_, loc=mu, scale=sigma)

    return v2, v1

d1 = TG(t(30.0764),t(1.4142),t(29.9000),t(100000.1016))

d2 = TG(t(31), t(1.1000), t(29.9000), t(100000.1016))

d3 = TG(t(30.05), t(1.4), t(29.9000), t(100000.1016))

print("KL(d1|d2)", kl_divergence(d1,d2))
print("KL(d2|d1)", kl_divergence(d2,d1))
print("KL(d1|d1)", kl_divergence(d1,d1))
print("KL(d2|d2)", kl_divergence(d2,d2))

print("KL(d1|d3)", kl_divergence(d1,d3))


# I think KL formula is wrong due to variance miscalculation

print(" ---- Variance ---- ")
print(calculate_var(30.0764,1.4142,29.9000,100000.1016))
print(calculate_var(31,1.1000,29.9000,100000.1016))