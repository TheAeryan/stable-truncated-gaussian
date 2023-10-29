# Simple script to test the stable implementation of icdf and rsample

from scipy.stats import truncnorm
from parallel_trunc_gaussian import ParallelTruncatedGaussian as TG
import torch
from torch import tensor as t
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

mu = 0
sigma = 1
a = -2
b = 2

# prob(x) given by the original truncated gaussian N(x|mu,sigma,a,b)
def prob_tn(a,b,x):
    return math.exp(TG(mu=t(0),sigma=t(1),a=t(a),b=t(b)).log_prob(t(x)).item())

# prob(x) of an exponential distribution exp(lambda=param)
def prob_exp(a, b, x):
    # We normalize x as the offset from a
    x = x-a
    
    # lambda=a
    val1 = a*math.exp(-a*x)

    # lambda equals N(x=a|0,1,a,b)
    # param = (2 / math.sqrt(2*math.pi))*( math.exp(-a**2/2) / (math.erf(b/math.sqrt(2))-math.erf(a/math.sqrt(2))) )
    # Directly calculating the expression above is unstable, so we simply use the method provided by TG
    param = math.exp(TG(mu=t(0),sigma=t(1),a=t(a),b=t(b)).log_prob(t(a)).item())  
    val2 = param*math.exp(-param*x)

    # return val1, val2
    # val2 works better than val1
    return val2

# We return prob(x**2)
def sqr_prob_exp(a, b, x):
    # We normalize x as the offset from a and square it
    x = (x-a)**2
    
    # lambda=a
    val1 = a*math.exp(-a*x)

    # lambda equals N(x=a|0,1,a,b)
    # param = (2 / math.sqrt(2*math.pi))*( math.exp(-a**2/2) / (math.erf(b/math.sqrt(2))-math.erf(a/math.sqrt(2))) )
    # Directly calculating the expression above is unstable, so we simply use the method provided by TG
    param = math.exp(TG(mu=t(0),sigma=t(1),a=t(a),b=t(b)).log_prob(t(a)).item())  
    val2 = param*math.exp(-param*x)

    # return val1, val2
    # val2 works better than val1
    return val2

# stable icdf of TN approximating its tail as an exponential distribution
# @perc The percentile, a number between 0 and 1
def stable_icdf(a,b,perc):
    assert perc>=0 and perc<=1

    if a>=0 and b>=0:
        if perc == 1: # We need this extreme case because, otherwise, we get log(0)
            return b

        # Calculate lambda parameter of the distribution
        # It is equal to N(x=a|mu,sigma,a,b)
        _lambda = math.exp(TG(mu=t(0),sigma=t(1),a=t(a),b=t(b)).log_prob(t(a)).item())

        # We add "a" to the result because, otherwise, result would be given
        # as an offset of a (and we want to obtain the absolute value in the number line)
        result = a + (-math.log(1-perc) / _lambda) 

        # We need to make sure that result is never larger than b
        clip_result = min(result, b)

    # Same as the previous case, but the exp distribution now goes from b to a (instead of from a to b)
    elif a<=0 and b<=0:
        if perc == 0: # We need this extreme case because, otherwise, we get log(0)
            return a
        
        # Calculate lambda parameter of the distribution
        # It is equal to N(x=b|mu,sigma,a,b)
        _lambda = math.exp(TG(mu=t(0),sigma=t(1),a=t(a),b=t(b)).log_prob(t(b)).item())  

        # The result is given as a negative offset from b
        # result = b - (-math.log(1-(1-perc)) / _lambda)
        result = b + math.log(perc) / _lambda # Equivalent to the line commented above

        # We need to make sure that result is never larger than b
        clip_result = max(result, a)

    else:
        raise Exception("Right now, we assume that a and b have the same sign")

    return clip_result


def stable_icdf_mu_sigma(mu,sigma,a,b,perc):
    alpha = (a-mu)/sigma
    beta = (b-mu)/sigma

    normalized_result = stable_icdf(a=alpha,b=beta,perc=perc)

    result = normalized_result*sigma + mu

    return result


# Same as the function above, but using scipy
def scipy_icdf(a,b,perc):
    return truncnorm.ppf(q=perc, a=a, b=b, loc=0, scale=1)

def scipy_icdf_mu_sigma(mu,sigma,a,b,perc):
    a_, b_ = (a - mu) / sigma, (b - mu) / sigma
    result = truncnorm.ppf(q=perc, a=a_, b=b_, loc=mu, scale=sigma)

    return result


# It compares the probabilities obtained using TN and the exp approximation
def compare_probs(a,b):
    # Generate x values from a to 2*a
    x_values = np.linspace(a, b, 1000)  # Adjust the number of points as needed

    # Calculate corresponding y values for both functions
    y_prob_tn = [prob_tn(a, b, x) for x in x_values]
    y_prob_exp = [prob_exp(a, b, x) for x in x_values]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_prob_tn, linewidth=1, label='prob_tn(a, b, x)')
    plt.plot(x_values, y_prob_exp, linewidth=1, label='prob_exp(a, b, x)')
    plt.xlabel('x')
    plt.ylabel('Probability')
    plt.title('Comparison of prob_tn and prob_exp Functions')
    plt.legend()
    plt.grid(True)
    plt.show()  

# It compares the icdf of the TN obtained "directly" (using scipy) and with our exp approximation
def compare_icdfs(a,b):
    # Generate x values from 0 to 1
    x_values = np.linspace(0, 1, 1000)  # Adjust the number of points as needed

    # Calculate corresponding y values for both functions
    y_stable = [stable_icdf(a, b, perc=x) for x in x_values]
    y_scipy = [scipy_icdf(a, b, perc=x) for x in x_values]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_stable, linewidth=1, label='My implementation')
    plt.plot(x_values, y_scipy, linewidth=1, label='Scipy')
    plt.xlabel('Percentile')
    plt.ylabel('Value')
    plt.title(f"Comparison of my icdf and scipy's with a={a} and b={b}")
    plt.legend()
    plt.grid(True)
    plt.show()

def compare_icdfs_mu_sigma(mu,sigma,a,b):
    # Generate x values from 0 to 1
    x_values = np.linspace(0, 1, 1000)  # Adjust the number of points as needed

    # Calculate corresponding y values for both functions
    y_stable = [stable_icdf_mu_sigma(mu, sigma, a, b, perc=x) for x in x_values]
    y_scipy = [scipy_icdf_mu_sigma(mu, sigma, a, b, perc=x) for x in x_values]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_stable, linewidth=1, label='My implementation')
    plt.plot(x_values, y_scipy, linewidth=1, label='Scipy')
    plt.xlabel('Percentile')
    plt.ylabel('Value')
    plt.title(f"Comparison of my icdf and scipy's with mu={mu}, sigma={sigma}, a={a} and b={b}")
    plt.legend()
    plt.grid(True)
    plt.show()   

# Like the function above, but now it uses the icdf method from TruncatedGaussian to
# obtain the icdf
def compare_icdfs_mu_sigma_list(mu, sigma, a, b):
    num_points = 1000

    # Obtain icdf using TG
    perc_t = torch.linspace(0, 1, steps=num_points)
    # We use the same mu,sigma,a,b values for each percentile in perc_tensor
    mu_t, sigma_t, a_t, b_t = t([mu]*num_points), t([sigma]*num_points), t([a]*num_points), t([b]*num_points)
    icdfs_tg = TG(mu_t, sigma_t, a_t, b_t).icdf(perc_t).tolist()

    # Obtain icdf using scipy
    a_, b_ = (a - mu) / sigma, (b - mu) / sigma # Scipy assumes a,b are given in relative units, as N times sigma from mu
    perc_l = np.linspace(0, 1, num_points)
    icdfs_scipy = [truncnorm.ppf(q=p, a=a_, b=b_, loc=mu, scale=sigma) for p in perc_l]

    #print("TG icdfs:", icdfs_tg)
    #print("Scipy icdfs:", icdfs_scipy)
    print(icdfs_tg[1], icdfs_tg[-2])

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(perc_l, icdfs_tg, linewidth=1, label='My implementation')
    plt.plot(perc_l, icdfs_scipy, linewidth=1, label='Scipy')
    plt.xlabel('Percentile')
    plt.ylabel('Value')
    plt.title(f"Comparison of TG and scipy's icdf with mu={mu}, sigma={sigma}, a={a} and b={b}")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__=='__main__':
    compare_icdfs_mu_sigma_list(mu, sigma, a, b)
    # compare_icdfs(a,b)
    #compare_icdfs_mu_sigma(mu,sigma,a,b)
    #print(stable_icdf(a,b,0.8))
    #compare_probs(a,b)



"""
# print("Scipy log-prob:", truncnorm.logpdf(a, a=a, b=b, loc=0, scale=1))
print("TN prob:", prob_tn(a,b,x))
print("Exp prob:", prob_exp(a,b,x))
print("Square exp prob:", sqr_prob_exp(a,b,x))
"""

"""
- Look at TN formula to see if lambda should be equal to a or something else
    - Its better to use lambda=N(a|mu,sigma,a,b) instead of lambda=a
- See if I should use exp(x) or exp(x^2) -> Compare probs of TN and exp in the vicinity of a
    - exp(x^2) works worse!!!

The exp is a good approximation to TN as long as a>>mu and the interval [a,b] is not too small
    
- Once we a good approximation of TN with exp, then obtain the icdf of exp.
  Add "a" to the result of icdf(exp(x)).
  Clip result between a and b.

Our approximation is good in most cases (mu far from [a,b] interval and [a,b] interval not too small)
Another issue is that we sometimes need to clip values for percentiles close to 0 (when a,b<0) or
close to 1 (when a,b>0)

- See how to normalize a,b,x when mu!=0 and sigma!=1

- Done

- See how large/small alpha,beta need to be in order to use this approximation instead of the original
  icdf formula
- Implement using tensors for several values at the same time
- Answer stack exchange post

"""