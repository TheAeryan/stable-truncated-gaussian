from parallel_trunc_gaussian import ParallelTruncatedGaussian as TG
import torch
from torch import tensor as t
from torch.distributions.kl import kl_divergence
from scipy.stats import truncnorm, entropy
import numpy as np

def calculate_var(mu, sigma, a, b):
    v1 = TG(t(mu), t(sigma), t(a), t(b)).variance

    a_, b_ = (a - mu) / sigma, (b - mu) / sigma
    v2 = truncnorm.var(a=a_, b=b_, loc=mu, scale=sigma)

    return v2, v1

def kl_approx(d1, d2):
    mu1, sigma1, a1, b1 = d1.mu, d1.sigma, d1.a, d1.b
    mu2, sigma2, a2, b2 = d2.mu, d2.sigma, d2.a, d2.b

    a1_, b1_ = (a1 - mu1) / sigma1, (b1 - mu1) / sigma1
    a2_, b2_ = (a2 - mu2) / sigma2, (b2 - mu2) / sigma2

    d1_scipy = truncnorm(a=a1_, b=b1_, loc=mu1, scale=sigma1)
    d2_scipy = truncnorm(a=a2_, b=b2_, loc=mu2, scale=sigma2)

    x = np.linspace(d2.a, d2.b, 100000)

    pdf1 = d1_scipy.pdf(x)
    pdf2 = d2_scipy.pdf(x)
    return entropy(pdf1, pdf2)

d1 = TG(t(30.0764),t(1.4142),t(29.9000),t(100000.1016))
d2 = TG(t(31), t(1.1000), t(29.9000), t(100000.1016))
d3 = TG(t(30.05), t(1.4), t(29.9000), t(100000.1016))

d4 = TG(t(3),t(1),t(-5),t(10000))
d5 = TG(t(4),t(1.5),t(-5),t(10000))

print("KL(d1|d2)", kl_divergence(d1,d2))
print("<Scipy> KL(d1|d2)", kl_approx(d1,d2))


print("KL(d2|d1)", kl_divergence(d2,d1))
print("KL(d1|d1)", kl_divergence(d1,d1))
print("KL(d2|d2)", kl_divergence(d2,d2))
print("KL(d1|d3)", kl_divergence(d1,d3))

print("KL(d4|d5)", kl_divergence(d4,d5))
print("KL(d5|d4)", kl_divergence(d5,d4))

print(" ---- Variance ---- ")

vars1 = calculate_var(30.0764,1.4142,29.9000,100000.1016)
vars2 = calculate_var(31,1.1000,29.9000,100000.1016)

print(torch.abs(vars1[0]-vars1[1]))
print(torch.abs(vars2[0]-vars2[1]))

# TODO
"""
- Compare results with scipy's kl divergence
- See if the KL divergence is wrong due to variance calculations

"""