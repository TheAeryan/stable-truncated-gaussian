# Tests for truncated gaussian
# I compare the values obtained with my code against those returned by scipy.stats.truncnorm

from parallel_trunc_gaussian import ParallelTruncatedGaussian as TG
from scipy.stats import truncnorm, entropy
import numpy as np
from torch import tensor as t
from torch.distributions.kl import kl_divergence
import math

"""
Current results:
    - Variance calculation seems to differ sometimes with scipy.
    - Mean calculation is the same as in scipy, except for mu:10, sigma:100, a:-1, b:1, where
      TG returns 4.100799560546875e-05 and scipy returns 0.00033332886587800203.
      Nonetheless, both values are very close to 0, so I don't think this is an issue
    - Log prob calculations are the same for scipy and TG

    --- KL-divergence
    It often performs well, but there are some errors:
        >>> Scipy:1.8040149237108782, TN:14.501136779785156 - params:((0, 5, -1, 1), (0, 0.1, -1, 1))
        - Scipy:0.6255919278246618, TN:0.7195351123809814 - params:((4, 1.5, -5, 10000), (3, 1, -5, 10000))
        - Scipy:7.861164322976255e-05, TN:-4.57763671875e-05 - params:((100, 1, 10, 20), (99, 1, 10, 20))

    It seems that these errors are not due to approximation errors in the variance calculation, mean calculation or log_Z calculation.
    Maybe the formula used to calculate the KL is not correct?
"""

# Two values are considered equal if their relative difference is less or equal than REL_TOL
REL_TOL = 0.01 # 1%

# Values for testing the mean and val
# In order: mu, sigma, a, b
mean_and_val_values = [
    [0,1,-1,1],
    [-1,1,-1,1],
    [-10,1,-1,1],
    [1,1,-1,1],
    [10,1,-1,1],
    [10,100,-1,1],
    [10,0.01,-1,1],

    [2.5,1,2,3],
    [2,1,2,3],
    [3,1,2,3],
    [0,1,2,3],
    [-100,1,2,3],
    [100,1,2,3],
    [100,0.01,2,3],
    [100,100,2,3],

    [20,1,20,2000],
    [2000,1,20,2000],
    [30,1,20,2000],
    [1990,1,20,2000],
    [10000,1,20,2000],
    [-10000,1,20,2000],
    [-10000,100,20,2000],

    [999.5,1,999,1000],
    [999,1,999,1000],
    [1000,1,999,1000],
    [3.5,1,999,1000],
    [1130.8,1,999,1000],

    [-999.5,1,-1000,-999],
    [-999,1,-1000,-999],
    [-1000,1,-1000,-999],
    [-3.5,1,-1000,-999],
    [-1130.8,1,-1000,-999],
]

# Values for testing the log_prob
# In order: x, mu, sigma, a, b
log_prob_values = [
    [0,0,1,-1,1],
    [0.5,-1,1,-1,1],
    [1,-10,1,-1,1],
    [-1,1,1,-1,1],
    [0.01,10,1,-1,1],
    [0.99,10,100,-1,1],
    [-0.84,10,0.01,-1,1],

    [2,2.5,1,2,3],
    [3,2,1,2,3],
    [2.5,3,1,2,3],
    [2.001,0,1,2,3],
    [2.999,-100,1,2,3],
    [2.489,100,1,2,3],
    [2,100,0.01,2,3],
    [2.846,100,100,2,3],

    [2000,20,1,20,2000],
    [20,2000,1,20,2000],
    [20.001,30,1,20,2000],
    [1999.99,1990,1,20,2000],
    [1000,10000,1,20,2000],
    [1691.56,-10000,1,20,2000],
    [23.19,-10000,100,20,2000],

    [999,999.5,1,999,1000],
    [1000,999,1,999,1000],
    [999.1,1000,1,999,1000],
    [999.5,3.5,1,999,1000],
    [999.99,1130.8,1,999,1000],

    [-999,-999.5,1,-1000,-999],
    [-1000,-999,1,-1000,-999],
    [-999.1,-1000,1,-1000,-999],
    [-999.999,-3.5,1,-1000,-999],
    [-999.45,-1130.8,1,-1000,-999]
]

# Values for testing the icdf (and, thus, rsample)
# In order: percentile, mu, sigma, a, b
icdf_values = [
    [0,0,1,-1,1],
    [0.5,-1,1,-1,1],
    [1,-10,1,-1,1],
    [0.2,1,1,-1,1],
    [0.01,10,1,-1,1],
    [0.99,10,100,-1,1],
    [0.3894,10,0.01,-1,1],

    [0.1,2.5,1,2,3],
    [0.2,2,1,2,3],
    [0.3,3,1,2,3],
    [0.4,0,1,2,3],
    [0.5,-100,1,2,3],
    [0.6,100,1,2,3],
    [0.7,100,0.01,2,3],
    [0.8,100,100,2,3],

    [0.9,20,1,20,2000],
    [1,2000,1,20,2000],
    [0.9999,30,1,20,2000],
    [0.0001,1990,1,20,2000],
    [0.99999,10000,1,20,2000],
    [0.00001,-10000,1,20,2000],
    [0.4598,-10000,100,20,2000],

    [0.1992,999.5,1,999,1000],
    [0.1446,999,1,999,1000],
    [0.78913,1000,1,999,1000],
    [0.4113,3.5,1,999,1000],
    [0.1617,1130.8,1,999,1000],

    [0.0146,-999.5,1,-1000,-999],
    [0.981616,-999,1,-1000,-999],
    [0.16464,-1000,1,-1000,-999],
    [0.89131,-3.5,1,-1000,-999],
    [0.16566,-1130.8,1,-1000,-999]
]

# Values for testing the KL
# The first element of each tuple is the first distribution, the second element is the second distribution
# Each distribution has parameters corresponding to mu,sigma,a,b
kl_values=[
    ((30, 1, 29, 100000), (31, 1, 29, 100000)),
    ((30, 1, 29, 100000), (30, 2, 29, 100000)),
    ((29, 1, 29, 100000), (30, 2, 29, 100000)),
    ((29, 1, 29, 100000), (30, 2, 20, 100000)),
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

    ((8, 1, 10, 20), (8, 1, 10, 20)),
    ((8, 1, 10, 20), (8, 1, 10, 100)),
    ((8, 1, 10, 20), (8, 1, -10, 20)),
    ((8, 1, 10, 20), (8, 1, -10, 100)),

    # Varying means with the same sigma and interval
    ((5, 1, 4, 100), (6, 1, 4, 100)),
    ((10, 2, 9, 500), (12, 2, 9, 500)),
    ((-5, 1, -6, 50), (-4, 1, -6, 50)),
    
    # Varying sigma with the same mean and interval
    ((5, 1, 4, 100), (5, 2, 4, 100)),
    ((10, 2, 9, 500), (10, 4, 9, 500)),
    ((-5, 1, -6, 50), (-5, 3, -6, 50)),
    
    # Varying interval with the same mean and sigma
    ((5, 1, 4, 10), (5, 1, 4, 50)),
    ((10, 2, 9, 40), (10, 2, 9, 500)),
    ((-5, 1, -20, -4), (-5, 1, -30, -4)),
    
    # Different means, sigmas, and intervals
    ((10, 2, 9, 40), (-5, 3, -20, 50)),
    
    # Boundary cases
    ((0, 1, -1, 1), (0, 2, -2, 2)),
    
    # Large vs small variance
    ((0, 0.1, -1, 1), (0, 5, -1, 1)),
    ((0, 5, -1, 1), (0, 0.1, -1, 1)),
    
    # Far away means
    ((-1000, 5, -1005, -995), (1000, 5, -1010, 1005)),
    ((-100, 2, -105, -95), (100, 2, -110, 105)),
    
    # Close means, far away intervals
    ((5, 1, 4, 6), (5, 1, 4, 96)),
    ((10, 2, 9, 11), (10, 2, 9, 91)),
    ((-5, 1, -6, -4), (-5, 1, -56, -4)),
    
    # mu is below the interval [a, b]
    ((-100, 1, 10, 20), (-99, 1, 10, 20)),
    ((-50, 2, 0, 10), (-49, 2, 0, 10)),
    ((-10, 3, 5, 15), (-9, 3, 5, 15)),
    
    # mu is above the interval [a, b]
    ((100, 1, 10, 20), (99, 1, 10, 20)),
    ((50, 2, 0, 10), (49, 2, 0, 10)),
    ((10, 3, -15, -5), (9, 3, -15, -5)),
    
    # Large vs small variance with mu outside the interval
    ((-100, 0.1, 10, 20), (-100, 5, 10, 20)),
    ((50, 0.1, 0, 10), (50, 5, 0, 10)),
    ((10, 0.1, -15, -5), (10, 5, -15, -5)),
    
    # Varying interval sizes with mu outside
    ((-100, 1, 10, 11), (-100, 1, 10, 50)),
    ((50, 2, 0, 1), (50, 2, 0, 20)),
    ((10, 3, -5, -4), (10, 3, -5, 15)),
]

# Returns the mean obtained with scipy and then the mean obtained with my code
def calculate_mean(mu, sigma, a, b):
    m1 = TG(t(mu), t(sigma), t(a), t(b)).mean

    a_, b_ = (a - mu) / sigma, (b - mu) / sigma
    m2 = truncnorm.mean(a=a_, b=b_, loc=mu, scale=sigma)

    return m2, m1

def calculate_var(mu, sigma, a, b):
    v1 = TG(t(mu), t(sigma), t(a), t(b)).variance

    a_, b_ = (a - mu) / sigma, (b - mu) / sigma
    v2 = truncnorm.var(a=a_, b=b_, loc=mu, scale=sigma)

    return v2, v1

def calculate_log_prob(x, mu, sigma, a, b):
    p1 = TG(t(mu), t(sigma), t(a), t(b)).log_prob(t(x))

    a_, b_ = (a - mu) / sigma, (b - mu) / sigma
    p2 = truncnorm.logpdf(x, a=a_, b=b_, loc=mu, scale=sigma)

    return p2, p1

def calculate_icdf(percentile, mu, sigma, a, b):
    icdf1 = TG(t(mu), t(sigma), t(a), t(b)).icdf(t(percentile))

    a_, b_ = (a - mu) / sigma, (b - mu) / sigma
    icdf2 = truncnorm.ppf(q=percentile, a=a_, b=b_, loc=mu, scale=sigma)

    return icdf2, icdf1

def check_mean_and_var():
    for mu, sigma, a, b in mean_and_val_values:
        t_mean, p_mean = calculate_mean(mu, sigma, a, b)
        t_var, p_var = calculate_var(mu, sigma, a, b)

        if not math.isclose(t_mean, p_mean, rel_tol=REL_TOL):
            print(f"Different means - Scipy:{t_mean}, TN:{p_mean} - mu:{mu}, sigma:{sigma}, a:{a}, b:{b}")

        if not math.isclose(t_var, p_var, rel_tol=REL_TOL):
            print(f"Different vars - Scipy:{t_var}, TN:{p_var} - mu:{mu}, sigma:{sigma}, a:{a}, b:{b}")

def check_log_prob():
    for x, mu, sigma, a, b in log_prob_values:
        t_p, p_p = calculate_log_prob(x, mu, sigma, a, b)

        if not math.isclose(t_p, p_p, rel_tol=REL_TOL):
            print(f"Different log probs - Scipy:{t_p}, TN:{p_p} - x:{x}, mu:{mu}, sigma:{sigma}, a:{a}, b:{b}")

def check_icdf():
    for percentile, mu, sigma, a, b in icdf_values:
        t_icdf, p_icdf = calculate_icdf(percentile, mu, sigma, a, b)

        if not math.isclose(t_icdf, p_icdf, rel_tol=REL_TOL):
            print(f"Different icdfs - Scipy:{t_icdf}, TN:{p_icdf} - percentile:{percentile}, mu:{mu}, sigma:{sigma}, a:{a}, b:{b}")

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

def check_kl():
    for params in kl_values:
        kl_scipy, kl = compare_kl(params[0], params[1])

        if not math.isclose(kl_scipy, kl, rel_tol=REL_TOL):
            print(f"Different KLs - Scipy:{kl_scipy}, TN:{kl} - params:{params}")


if __name__ == '__main__':
    #check_mean_and_var()
    #check_log_prob()
    #check_icdf()
    check_kl()
