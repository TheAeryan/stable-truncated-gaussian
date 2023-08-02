# Tests for truncated gaussian
# I compare the values obtained with my code against those returned by scipy.stats.truncnorm

from parallel_trunc_gaussian import ParallelTruncatedGaussian as TG
from scipy.stats import truncnorm
from torch import tensor as t
import math

"""
Current results:
    - Variance calculation seems to differ sometimes with scipy.
    - Mean calculation is the same as in scipy, except for mu:10, sigma:100, a:-1, b:1, where
      TG returns 4.100799560546875e-05 and scipy returns 0.00033332886587800203.
      Nonetheless, both values are very close to 0, so I don't think this is an issue
    - Log prob calculations are the same for scipy and TG
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

if __name__ == '__main__':
    #check_mean_and_var()
    #check_log_prob()
    check_icdf()
