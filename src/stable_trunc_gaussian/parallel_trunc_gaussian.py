"""
---- Stable Truncated Gaussian ----

Implementation of truncated gaussian which results numerically stable even when the mu parameter is outside the
interval [a,b] given by the bounds.
In order to obtain this implementation, we have employed the formulas detailed on https://github.com/cossio/
TruncatedNormal.jl/blob/23bfc7d0189ca6857e2e498006bbbed2a8b58be7/notes/normal.pdf
Our implementation is also inspired by torch_truncnorm: https://github.com/toshas/torch_truncnorm/blob/main/TruncatedNormal.py#L40
"""

import torch
from torch.special import erf, erfc, erfcx, erfinv
from torch.distributions import Distribution, constraints
from torch.distributions.kl import register_kl
from torch import where
from torch import logical_and as t_and, logical_or as t_or, logical_not as t_not
import math

# Constants
SQRT_PI = math.sqrt(math.pi)
SQRT_2 = math.sqrt(2)
SQRT_2_PI = math.sqrt(2*math.pi)
INV_SQRT_2 = 1/SQRT_2
INV_SQRT_PI = 1/SQRT_PI
INV_PI = 1/math.pi
SQRT_2_DIV_SQRT_PI = SQRT_2 / SQRT_PI
LOG_SQRT_2_PI = math.log(SQRT_2_PI)
LOG_2 = math.log(2)

# Auxiliary function used to check which tensor values are "close" to some float
# We only use absolute tolerance but not relative one
# In other words, two values a, b are considered to be close if |a-b|<=atol
def is_close(tensor, val, atol):
    val_tensor = torch.full_like(tensor, val)
    return torch.abs(tensor-val_tensor) < atol

class ParallelTruncatedGaussian(Distribution):

    # Absolute tolerance used as "atol" parameter for is_close method
    # It is used in the icdf method, to check if "perc" is very close to 0 or 1
    # If it is, we return either a or b, respectively
    # A lower atol parameter may result in NaN values when calculating the icdf
    atol = 5e-5
    has_rsample = True

    arg_constraints = {
        'mu': constraints.real,
        'sigma': constraints.positive,
        'a': constraints.real,
        'b': constraints.real,
    }

    """
    Constructor
        @mu Mu parameter, corresponding to the mean of the gaussian <before> truncation
        @sigma Sigma parameter, corresponding to the standard deviation of the gaussian <before> truncation
        @a Lower bound
        @b Upper bound

        All parameters must be instances of torch.Tensor with a single value (i.e., be of dimension 0). 
    """
    def __init__(self, mu, sigma, a, b):
        # Safety checks
        if not (isinstance(mu, torch.Tensor) and isinstance(sigma, torch.Tensor) and isinstance(a, torch.Tensor) \
            and isinstance(b, torch.Tensor)):
            raise ValueError("All the parameters must be instances of torch.Tensor")	
        if torch.any(sigma <= 0):
            raise ValueError("sigma must be greater than 0")
        if torch.any(a >= b):
            raise ValueError("parameter 'a' (lower bound) must be smaller than 'b' (upper bound)")  

        self._mu = mu
        self._sigma = sigma
        self._a = a
        self._b = b
        self._alpha = (self._a-self._mu)/self._sigma
        self._beta = (self._b-self._mu)/self._sigma

        # Calculate the mean and var once and store them
        self._mean = self._calculate_mean()
        self._variance = self._calculate_variance()

        # Calculate log(Z) = log(big_phi(beta)-big_phi(alpha))
        self._log_Z = self._calculate_log_Z()
        self._Z = torch.exp(self._log_Z)

        # big_phi(alpha), used in the icdf (and rsample) methods
        self._big_phi_alpha = self._big_phi(self._alpha)

        batch_shape = a.size()
        super().__init__(batch_shape)

    # I override __repr__ because parameter names that appear in arg_constraints are of the form
    # 'param' but instance attributes are of the form '_param' 
    def __repr__(self) -> str:
        param_names = [k for k, _ in self.arg_constraints.items() if '_'+k in self.__dict__]
        args_string = ', '.join(['{}: {}'.format(p, self.__dict__['_'+p]
                                if self.__dict__['_'+p].numel() == 1
                                else self.__dict__['_'+p].size()) for p in param_names])
        return self.__class__.__name__ + '(' + args_string + ')'

    @property
    def mu(self):
        return self._mu
    
    @property
    def sigma(self):
        return self._sigma
    
    @property
    def a(self):
        return self._a
    
    @property
    def b(self):
        return self._b
    
    @property
    def alpha(self):
        return self._alpha
    
    @property
    def beta(self):
        return self._beta
    
    # --- Auxiliary methods ---

    @staticmethod
    def _big_phi(x):
        return 0.5*(1 + erf(INV_SQRT_2*x))

    @staticmethod
    def _inv_big_phi(x):
        return SQRT_2 * erfinv(2*x-1)

    @staticmethod
    def _delta(x, y):
        return torch.exp(x**2 - y**2)
    
    # Taylor expansion of F_1(x, eps=y-x)
    @staticmethod
    def _P_1(x, eps):
        t0 = SQRT_PI*x
        t1 = 0.5*SQRT_PI*eps
        t2 = -(1/6)*SQRT_PI*x*eps**2
        t3 = -(1/12)*SQRT_PI*eps**3
        t4 = (1/90)*SQRT_PI*x*(x**2+1)*eps**4
        return t0 + t1 + t2 + t3 + t4
    
    # Taylor expansion of F_2(x, eps=y-x)
    @staticmethod
    def _P_2(x, eps):
        t0 = 0.5*SQRT_PI*(2*x**2-1)
        t1 = SQRT_PI*x*eps
        t2 = -(1/3)*SQRT_PI*(x**2-1)*eps**2
        t3 = -(1/3)*SQRT_PI*x*eps**3
        t4 = (1/90)*SQRT_PI*(2*x**4+3*x**2-8)*eps**4
        return t0 + t1 + t2 + t3 + t4
    
    @staticmethod
    def _F_1(x_, y_):
        # All values in tensor @x must be smaller than values in tensor @y
        x = torch.where(torch.abs(x_)<=torch.abs(y_),x_,y_)
        y = torch.where(torch.abs(y_)>=torch.abs(x_),y_,x_)

        # Obtain masks
        with torch.no_grad():
            out1_cond = torch.abs(x - y) < 1e-7 # tensor([False, False, False, False])
            out2_cond = torch.logical_and( torch.logical_and(x<=0, y<=0), torch.logical_not(out1_cond) ) # tensor([False,  True, False, False])
            out3_cond = torch.logical_and( torch.logical_and(x>=0, y>=0), torch.logical_not(out1_cond) ) # tensor([False, False, False,  True]) 
            out4_cond = torch.logical_and( torch.logical_and( torch.logical_not(out1_cond), torch.logical_not(out2_cond) ),
                                           torch.logical_not(out3_cond) ) # All the other conditions must be false


        # Mask input values for each operation
        x1, y1 = where(out1_cond, x, 0), where(out1_cond, y, 0)
        x2, y2 = where(out2_cond, x, -1), where(out2_cond, y, 0)
        x3, y3 = where(out3_cond, x, 0), where(out3_cond, y, 1)
        x4, y4 = where(out4_cond, x, 0), where(out4_cond, y, 1)

        # Apply operations
        delt = ParallelTruncatedGaussian._delta(x, y)
        one_minus_delt = 1 - delt

        out1_m = ParallelTruncatedGaussian._P_1(x1, y1-x1)
        out2_m = one_minus_delt / (delt*erfcx(-y2) - erfcx(-x2))
        out3_m = one_minus_delt / (erfcx(x3) - delt*erfcx(y3))
        out4_m = (one_minus_delt*torch.exp(-x4**2)) / (erf(y4)-erf(x4))

        # Unmask tensors, by setting masked values to 0
        out1 = where(out1_cond, out1_m, 0)
        out2 = where(out2_cond, out2_m, 0)
        out3 = where(out3_cond, out3_m, 0)
        out4 = where(out4_cond, out4_m, 0)

        # Add them up into a single tensor
        final_out = out1 + out2 + out3 + out4
        
        return final_out

    @staticmethod
    def _F_2(x_, y_):
        # All values in tensor @x must be smaller than values in tensor @y
        x = torch.where(torch.abs(x_)<=torch.abs(y_),x_,y_)
        y = torch.where(torch.abs(y_)>=torch.abs(x_),y_,x_)

        # Obtain masks
        with torch.no_grad():
            out1_cond = torch.abs(x - y) < 1e-7 # tensor([False, False, False, False])
            out2_cond = torch.logical_and( torch.logical_and(x<=0, y<=0), torch.logical_not(out1_cond) ) # tensor([False,  True, False, False])
            out3_cond = torch.logical_and( torch.logical_and(x>=0, y>=0), torch.logical_not(out1_cond) ) # tensor([False, False, False,  True]) 
            out4_cond = torch.logical_and( torch.logical_and( torch.logical_not(out1_cond), torch.logical_not(out2_cond) ),
                                           torch.logical_not(out3_cond) ) # All the other conditions must be false

        # Mask input values for each operation
        x1, y1 = where(out1_cond, x, 0), where(out1_cond, y, 0)
        x2, y2 = where(out2_cond, x, -1), where(out2_cond, y, 0)
        x3, y3 = where(out3_cond, x, 0), where(out3_cond, y, 1)
        x4, y4 = where(out4_cond, x, 0), where(out4_cond, y, 1)

        # Apply operations
        delt = ParallelTruncatedGaussian._delta(x, y)
        x_minus_y_by_delt = x - y*delt

        out1_m = ParallelTruncatedGaussian._P_2(x1, y1-x1)
        out2_m = x_minus_y_by_delt / (delt*erfcx(-y2) - erfcx(-x2)) 
        out3_m = x_minus_y_by_delt / (erfcx(x3) - delt*erfcx(y3))
        out4_m = (x_minus_y_by_delt*torch.exp(-x4**2)) / (erf(y4)-erf(x4))

        # Unmask tensors, by setting masked values to 0
        out1 = where(out1_cond, out1_m, 0)
        out2 = where(out2_cond, out2_m, 0)
        out3 = where(out3_cond, out3_m, 0)
        out4 = where(out4_cond, out4_m, 0)

        # Add them up into a single tensor
        final_out = out1 + out2 + out3 + out4

        return final_out

    # --- Main methods ---

    # Numerically stable implementation of Z=log(big_phi(beta)-big_phi(alpha))
    def _calculate_log_Z(self):
        # torch.where autograd does not work correctly when there are NaN or inf values
        # (even if those values are not chosen by the condition in torch.where)
        # https://discuss.pytorch.org/t/incorrect-gradient-calculation-with-torch-where-and-nans/185367
        # Therefore, we need to avoid getting NaN or inf for ANY operation

        alpha, beta, mu, mean_d = self._alpha, self._beta, self._mu, self._mean.detach()

        # Obtain masks
        with torch.no_grad():
            out1_cond = torch.logical_and(alpha>=0, beta>=0)
            out2_cond = torch.logical_and(alpha<=0, beta<=0)
            out3_cond = torch.logical_and(torch.logical_not(out1_cond), torch.logical_not(out2_cond))

        # Mask input values for each operation
        alpha1, beta1, mu1 = where(out1_cond, alpha, 0), where(out1_cond, beta, 1), where(out1_cond, mu, mean_d-1)
        alpha2, beta2, mu2 = where(out2_cond, alpha, 1), where(out2_cond, beta, 0), where(out2_cond, mu, mean_d+1)
        alpha3, beta3 = where(out3_cond, alpha, 0), where(out3_cond, beta, 1)

        # Apply operations
        out1_m = -torch.log( (self._mean-mu1) / self._sigma ) - LOG_SQRT_2_PI - (alpha1**2)/2 + \
                torch.log(1 - torch.exp( (alpha1+beta1)*(alpha1-beta1) / 2 ))
        out2_m = -torch.log( (mu2-self._mean) / self._sigma ) - LOG_SQRT_2_PI - (beta2**2)/2 + \
                torch.log(1 - torch.exp( (alpha2+beta2)*(beta2-alpha2) / 2))
        out3_m = -LOG_2 + torch.log(erf(beta3*INV_SQRT_2) - erf(alpha3*INV_SQRT_2))

        # Unmask tensors, by setting masked values to 0
        out1 = where(out1_cond, out1_m, 0)
        out2 = where(out2_cond, out2_m, 0)
        out3 = where(out3_cond, out3_m, 0)

        # Add them up into a single tensor
        final_out = out1 + out2 + out3

        return final_out	
    
    @property
    def log_Z(self):
        return self._log_Z

    @property
    def Z(self):
        return self._Z

    # Mean of the distribution (after truncation)
    def _calculate_mean(self):
        cls = self.__class__
        fraction = SQRT_2_DIV_SQRT_PI * cls._F_1(self._alpha*INV_SQRT_2, self._beta*INV_SQRT_2)
        result = self._mu + fraction*self._sigma

        return result

    @property
    def mean(self):
        return self._mean
        
    # Variance of the distribution (after truncation)
    def _calculate_variance(self):
        cls = self.__class__
        fraction_1 = (2*INV_SQRT_PI)*cls._F_2(self._alpha*INV_SQRT_2, self._beta*INV_SQRT_2)
        fraction_2 = (2*INV_PI)*cls._F_1(self._alpha*INV_SQRT_2, self._beta*INV_SQRT_2)**2
        result = (1+fraction_1-fraction_2)*self._sigma**2

        return result
    
    @property
    def variance(self):
        return self._variance
    
    # Log probability of some value(s) x under the truncated gaussian
    # Unlike the parameters of the distributions, we allow x to be a tensor of arbitrary size
    # (e.g., x=torch.tensor([0,1,2]))
    def log_prob(self, x):
        # Check that x is a tensor
        if not isinstance(x, torch.Tensor):
            raise ValueError("parameter 'x' must be an instance of torch.Tensor") 

        # x must be inside the interval [a, b]
        if torch.any(x < self._a) or torch.any(x > self._b):
            raise ValueError(f"parameter 'x' ({x}) is outside the [a={self._a}, b={self._b}] interval")

        xi = (x-self._mu)/self._sigma
        term_1 = -torch.log(self._sigma)
        term_2 = -LOG_SQRT_2_PI - (xi**2)/2
        term_3 = -self._log_Z
        result = term_1 + term_2 + term_3

        return result

    # Inverse cdf function or percentile function (ppf)
    # @perc Percentile, a real number between 0 and 1 (both included)
    # NOTE: when perc is very close (according to self.atol) to either 0 or 1, we return a or b, respectively
    # In the future, we will improve the precision of icdf for values very close to 0 or 1
    def icdf(self, perc):
        # Due to numerical unstability, when computing icdf for large (positive or negative) alpha, beta
        # we don't use the "straightforward" formula but, rather, approximate the tail of the gaussian using
        # an exponential distribution
        # See: https://math.stackexchange.com/questions/4746255/numerically-stable-method-for-sampling-from-truncated-normal-distribution/4746917#4746917

        # This variable is used to decide when to use the "straightforward" formula for the
        # TN icdf and when to use the exp-based formula (which only works for the tails of the TN)
        threshold = 3

        if not isinstance(perc, torch.Tensor):
            raise ValueError("parameter 'perc' must be an instance of torch.Tensor") 

        alpha, beta, mu, sigma = self._alpha, self._beta, self._mu, self._sigma
        # Prob given by the N(x|mu,sigma,a,b) dist. to x=a and x=b
        # NOTE: We need to clip log_probs because, otherwise, prob_a and prob_b may
        # be 0 (which later results in "inf" values in calculations)
        # Since a log_prob of -30 means the probability is about 1e-13, in practice, this clipping
        # does not affect the final result. It only affects values which will nonetheless
        # be masked to 0, i.e., which do not affect the final result.
        log_prob_a = torch.clip(self.log_prob(self._a), min=-30, max=30)
        log_prob_b = torch.clip(self.log_prob(self._b), min=-30, max=30)
        prob_a, prob_b = torch.exp(log_prob_a), torch.exp(log_prob_b)

        # Obtain masks
        with torch.no_grad():
            out1_cond = is_close(perc, 0, atol=ParallelTruncatedGaussian.atol) 
            out2_cond = is_close(perc, 1, atol=ParallelTruncatedGaussian.atol)
            # right tail approximation
            out3_cond = t_and(alpha>=threshold, t_not(t_or(out1_cond, out2_cond)))
            # left tail approximation
            out4_cond = t_and(beta<=-threshold , t_not(t_or(out1_cond, out2_cond)))
            # straightforward formula
            out5_cond = t_not(t_or(t_or(t_or(out1_cond, out2_cond), out3_cond), out4_cond))

        # Mask input values for each operation
        # For those perc values that are either 0 or 1, why put them to 0.5 to avoid 
        # log(0) (note that this values are not used for the final result, since they are masked out)
        perc_3_4 = where(t_or(out1_cond,out2_cond), 0.5, perc)

        # For the stable formula, the input to _inv_big_phi must be 0.5 for masked values
        # Otherwise, we could get -inf (for 0) or inf (for 1)
        inv_big_phi_input = self._big_phi_alpha + perc*self._Z
        inv_big_phi_input_m = where(out5_cond, inv_big_phi_input, 0.5)

        # -- Truncated exponential icdf --
        # See https://math.stackexchange.com/questions/4746255/numerically-stable-method-for-sampling-from-truncated-normal-distribution/
        # The icdf of an exponential distribution lambda*e^(-lambda*x) is -ln(1-p)/lambda, where p is the percentile in [0,1]
        # We now want to obtain the icdf of a *truncated* exponential distribution in the range [0, gamma] (where gamma=beta-alpha)
        # Therefore, the icdf of this truncated exp dist should output a value between [0, gamma] (instead of in [0, inf) )
        # We can do this by using the same icdf function but modifying the percentile p
        # p_gamma=1-e^(-gamma*lambda) is the percentile for which icdf returns gamma. Therefore, the new percentile p' must be a number
        # between 0 and p_gamma, so that icdf(p') is inside [0, gamma]
        with torch.no_grad():
            gamma = beta-alpha
            p_gamma_a = 1 - torch.exp(-gamma*prob_a) # p_gamma for when lambda=prob_a
            p_gamma_b = 1 - torch.exp(-gamma*prob_b) # p_gamma for when lambda=prob_a

            # Scale percentiles from range [0,1] to [0,p_gamma]
            perc_3_scaled = perc_3_4*p_gamma_a
            perc_4_scaled = (1-perc_3_4)*p_gamma_b # We do 1-perc_3_4 because the percentile for the left tail approximation goes from
                                                   # 1 to 0 instead of from 0 to 1 (i.e., it's the opposite of the percentile for the right tail approx.)

        # Apply operations
        out1_m = self._a
        out2_m = self._b
        # Right tail approximation
        out3_m = (alpha + (-torch.log(1-perc_3_scaled) / prob_a))*sigma + mu
        # Left tail approximation
        out4_m = (beta + torch.log(1-perc_4_scaled) / prob_b)*sigma + mu
        # Straightforward formula
        # We clip the result because sometimes, inv_big_phi can return a value slightly smaller than alpha
        # or larger than beta (e.g., 3.9999 when alpha is 4)
        out5_m = (self._inv_big_phi( inv_big_phi_input_m ))*sigma + mu

        # Unmask tensors, by setting masked values to 0
        out1 = where(out1_cond, out1_m, 0)
        out2 = where(out2_cond, out2_m, 0)
        out3 = where(out3_cond, out3_m, 0)
        out4 = where(out4_cond, out4_m, 0)
        out5 = where(out5_cond, out5_m, 0)

        # Add them up into a single tensor
        final_out = out1 + out2 + out3 + out4 + out5

        # "Smart" clipping method that does not result in zero gradients when values outside the interval [a,b]
        # are clipped to lie inside this interval
        # We need to clip final_out due to approximation errors when calculating the icdf
        final_out_clip = final_out + torch.clip(self._a-final_out, min=0) - torch.clip(final_out-self._b, min=0)

        return final_out_clip


    """
    From https://pytorch.org/docs/stable/distributions.html
    Generates a sample_shape shaped reparameterized sample or sample_shape shaped batch of 
    reparameterized samples if the distribution parameters are batched.
    This sampling process is differentiable.
    Current rsample implementation extracted from https://github.com/toshas/torch_truncnorm

    @sample_shape The shape (e.g., how many samples) to generate for each combinations of parameters
                  mu,sigma,a,b
                  Example:
                  >>> output = TG(t([-20,10]),t([1,10]),t([-20,10]),t([-10,11])).rsample([3])
                  >>> print(output)
                  tensor([[-19.3958,  10.9927],
                          [-19.1791,  10.6582],
                          [-18.8327,  10.5413]])
                 
                  where output[i] contains sample_shape=3 samples, each one corresponding to a different
                  parameter combination (i.e., output[i][j] is the i-th sample of the j-th parameter combination)
    """
    # Note: the current implementation of icdf returns a or b for values very close to 0 or 1, respectively
    # To avoid this, instead of sampling values uniformly from [0,1], we sample them from [self.atol, 1-self.atol]
    def rsample(self, sample_shape: torch.Size = torch.Size()):
        shape = self._extended_shape(sample_shape)
        
        #p = torch.empty(shape, device=self._mu.device).uniform_(0,1)
        p = torch.empty(shape, device=self._mu.device).uniform_(ParallelTruncatedGaussian.atol,
                                                          1-ParallelTruncatedGaussian.atol)
        
        out = self.icdf(p)

        return out

"""
--- KL Divergence ---
Function for calculating the KL divergence between two Truncated Gaussian distributions.

We use the formulas detailed in the paper "Statistical Divergences between 
Densities of Truncated Exponential Families with Nested Supports: Duo Bregman
and Duo Jensen Divergences" by Frank Nielsen (see Eq. 111).

Don't call this function directly. Instead, do the following:

    from torch.distributions.kl import kl_divergence

    ...

    value = kl_divergence(trunc_gauss_1, trunc_gauss_2)
"""
@register_kl(ParallelTruncatedGaussian, ParallelTruncatedGaussian)
def kl_truncgauss_truncgauss(d1, d2):
    mu1, sigma1, a1, b1, log_Z1, mean1, var1 = d1.mu, d1.sigma, d1.a, d1.b, d1.log_Z, d1.mean, d1.variance
    mu2, sigma2, a2, b2, log_Z2, mean2, var2 = d2.mu, d2.sigma, d2.a, d2.b, d2.log_Z, d2.mean, d2.variance
    inv_sqr_sigma1 = 1/(sigma1**2)
    inv_sqr_sigma2 = 1/(sigma2**2)

    # The interval [a1, b1] must be inside the interval [a2, b2]
    # Otherwise, the KL divergence is infinite
    if torch.any(a1 < a2) or torch.any(b1 > b2):
        raise Exception(f"Interval [a1={a1}, b1={b1}] must be inside interval [a2={a2}, b2={b2}]. Otherwise, KL Divergence equals +inf.")

    # Calculate KL(d1 || d2)
    kl_divergence = 0.5*mu2*inv_sqr_sigma2 - 0.5*mu1*inv_sqr_sigma1 + torch.log(sigma2/sigma1) + log_Z2 - log_Z1 - \
                    (mu2*inv_sqr_sigma2 - mu1*inv_sqr_sigma1)*mean1 - (0.5*inv_sqr_sigma1 - 0.5*inv_sqr_sigma2)*(var1+mean1**2)

    return kl_divergence
