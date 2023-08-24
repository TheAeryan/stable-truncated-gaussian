# Simple script to test the stable implementation of icdf and rsample

from scipy.stats import truncnorm
from parallel_trunc_gaussian import ParallelTruncatedGaussian as TG
from torch import tensor as t
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

a = 20
b = 20.1

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

    if perc == 1: # We need this extreme case because, otherwise, we get log(0)
        return b

    # Calculate lambda parameter of the distribution
    # It is equal to N(x=a|mu,sigma,a,b)
    _lambda = math.exp(TG(mu=t(0),sigma=t(1),a=t(a),b=t(b)).log_prob(t(a)).item())  

    # We add "a" to the result because, otherwise, result would be given
    # as an offset of a (and we want to obtain the absolute value in the number line)
    result = (-math.log(1-perc) / _lambda) + a 

    # We need to make sure that result is never larger than b
    clip_result = min(result, b)

    return clip_result

# Same as the function above, but using scipy
def scipy_icdf(a,b,perc):
    return truncnorm.ppf(q=perc, a=a, b=b, loc=0, scale=1)

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




compare_icdfs(a,b)

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
- See how to normalize a,b,x when mu!=0 and sigma!=1
- See how large/small alpha,beta need to be in order to use this approximation instead of the original
  icdf formula
- Implement using tensors for several values at the same time
- Answer stack exchange post

"""