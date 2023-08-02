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


class ParallelTruncatedGaussian(Distribution):

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

	# Inverse cumulative distribution function
	def icdf(self, x):
		return self._inv_big_phi( self._big_phi_alpha + x*self._Z )*self._sigma + self._mu

	"""
	From https://pytorch.org/docs/stable/distributions.html
	Generates a sample_shape shaped reparameterized sample or sample_shape shaped batch of 
	reparameterized samples if the distribution parameters are batched.
	This sampling process is differentiable.
	Current rsample implementation extracted from https://github.com/toshas/torch_truncnorm

	Note: right now, the implementation of rsample is numerically unstable!!
	"""
	def rsample(self, sample_shape: torch.Size = torch.Size()):
		shape = self._extended_shape(sample_shape)
		
		p = torch.empty(shape, device=self._mu.device).uniform_(0,1)
		
		return self.icdf(p)

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
	if a1 < a2 or b1 > b2:
		raise Exception(f"Interval [a1={a1}, b1={b1}] must be inside interval [a2={a2}, b2={b2}]. Otherwise, KL Divergence equals +inf.")

	# Calculate KL(d1 || d2)
	kl_divergence = 0.5*mu2*inv_sqr_sigma2 - 0.5*mu1*inv_sqr_sigma1 + torch.log(sigma2/sigma1) + log_Z2 - log_Z1 - \
					(mu2*inv_sqr_sigma2 - mu1*inv_sqr_sigma1)*mean1 - (0.5*inv_sqr_sigma1 - 0.5*inv_sqr_sigma2)*(var1+mean1**2)

	return kl_divergence