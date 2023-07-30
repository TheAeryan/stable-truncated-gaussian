"""
---- Stable Truncated Gaussian ----

Implementation of truncated gaussian which results numerically stable even when the mu parameter is outside the
interval [a,b] given by the bounds.
In order to obtain this implementation, we have employed the formulas detailed on https://github.com/cossio/
TruncatedNormal.jl/blob/23bfc7d0189ca6857e2e498006bbbed2a8b58be7/notes/normal.pdf

Right now, we only implement the following methods: mean, variance and log_prob.
"""

import torch
from torch.special import erf, erfc, erfcx
from torch.distributions import Distribution, constraints
import math

# Constants
SQRT_PI = math.sqrt(math.pi)
SQRT_2 = math.sqrt(2)
SQRT_2_PI = math.sqrt(2*math.pi)
INV_SQRT_2 = 1/SQRT_2
INV_SQRT_PI = 1/SQRT_PI
INV_PI = 1/math.pi
LOG_SQRT_2_PI = math.log(SQRT_2_PI)
LOG_2 = math.log(2)

class ParallelTruncatedGaussian(Distribution):

	has_rsample = False

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

		# Values
		delt = ParallelTruncatedGaussian._delta(x, y)
		one_minus_delt = 1 - delt
		out1 = ParallelTruncatedGaussian._P_1(x, y-x)
		out2 = one_minus_delt / (delt*erfcx(-y) - erfcx(-x))
		out3 = one_minus_delt / (erfcx(x) - delt*erfcx(y))
		out4 = (one_minus_delt*torch.exp(-x**2)) / (erf(y)-erf(x))

		# Conditions
		out1_cond = torch.abs(x - y) < 1e-7
		out2_cond = torch.logical_and( torch.logical_and(x<=0, y<=0), torch.logical_not(out1_cond) )
		out3_cond = torch.logical_and( torch.logical_and(x>=0, y>=0), torch.logical_not(out1_cond) )
		out4_cond = torch.logical_and( torch.logical_and( torch.logical_not(out1_cond), torch.logical_not(out2_cond) ),
									   torch.logical_not(out3_cond) ) # All the other conditions must be false

		# Final expression
		# We need to set nan, -inf, inf values to 0.0 since, otherwise, 0*inf may result in NaNs
		# in the final formula
		final_out = torch.nan_to_num(out1,nan=0.0,posinf=0.0,neginf=0.0)*out1_cond + \
			  		torch.nan_to_num(out2,nan=0.0,posinf=0.0,neginf=0.0)*out2_cond + \
					torch.nan_to_num(out3,nan=0.0,posinf=0.0,neginf=0.0)*out3_cond + \
					torch.nan_to_num(out4,nan=0.0,posinf=0.0,neginf=0.0)*out4_cond

		return final_out

		"""
		Sequential code
	
		if torch.abs(x) > torch.abs(y):
			out = ParallelTruncatedGaussian._F_1(y, x)

		elif torch.abs(x - y) < 1e-7: #out1_cond
			out = ParallelTruncatedGaussian._P_1(x, y-x)
		elif x <= 0 and y <= 0: #out2_cond
			delt = ParallelTruncatedGaussian._delta(x, y)
			out = (1 - delt) / (delt*erfcx(-y) - erfcx(-x))  
		elif x >= 0 and y >= 0: #out3_cond
			delt = ParallelTruncatedGaussian._delta(x, y)
			out = (1 - delt) / (erfcx(x) - delt*erfcx(y))
		else: #out4_cond
			delt = ParallelTruncatedGaussian._delta(x, y)
			out = ((1-delt)*torch.exp(-x**2)) / (erf(y)-erf(x))
		"""
	
	@staticmethod
	def _F_2(x_, y_):
		# All values in tensor @x must be smaller than values in tensor @y
		x = torch.where(torch.abs(x_)<=torch.abs(y_),x_,y_)
		y = torch.where(torch.abs(y_)>=torch.abs(x_),y_,x_)

		# Values
		delt = ParallelTruncatedGaussian._delta(x, y)
		x_minus_y_by_delt = x - y*delt
		out1 = ParallelTruncatedGaussian._P_2(x, y-x)
		out2 = x_minus_y_by_delt / (delt*erfcx(-y) - erfcx(-x)) 
		out3 = x_minus_y_by_delt / (erfcx(x) - delt*erfcx(y))
		out4 = (x_minus_y_by_delt*torch.exp(-x**2)) / (erf(y)-erf(x))

		# Conditions
		out1_cond = torch.abs(x - y) < 1e-7
		out2_cond = torch.logical_and( torch.logical_and(x<=0, y<=0), torch.logical_not(out1_cond) )
		out3_cond = torch.logical_and( torch.logical_and(x>=0, y>=0), torch.logical_not(out1_cond) )
		out4_cond = torch.logical_and( torch.logical_and( torch.logical_not(out1_cond), torch.logical_not(out2_cond) ),
									   torch.logical_not(out3_cond) ) # All the other conditions must be false

		# We need to set nan, -inf, inf values to 0.0 since, otherwise, 0*inf may result in NaNs
		# in the final formula
		final_out = torch.nan_to_num(out1,nan=0.0,posinf=0.0,neginf=0.0)*out1_cond + \
			  		torch.nan_to_num(out2,nan=0.0,posinf=0.0,neginf=0.0)*out2_cond + \
					torch.nan_to_num(out3,nan=0.0,posinf=0.0,neginf=0.0)*out3_cond + \
					torch.nan_to_num(out4,nan=0.0,posinf=0.0,neginf=0.0)*out4_cond

		return final_out

		"""
		Sequential code
		
		if torch.abs(x) > torch.abs(y):
			out = ParallelTruncatedGaussian._F_2(y, x)

		elif torch.abs(x - y) < 1e-7: #out1_cond
			out = ParallelTruncatedGaussian._P_2(x, y-x)
		elif x <= 0 and y <= 0: #out2_cond
			delt = ParallelTruncatedGaussian._delta(x, y)
			out = (x - y*delt) / (delt*erfcx(-y) - erfcx(-x))  
		elif x >= 0 and y >= 0: #out3_cond 
			delt = ParallelTruncatedGaussian._delta(x, y)
			out = (x - y*delt) / (erfcx(x) - delt*erfcx(y))
		else: #out4_cond
			delt = ParallelTruncatedGaussian._delta(x, y)
			out = ((x-y*delt)*torch.exp(-x**2)) / (erf(y)-erf(x))
		"""
		
	
	# Numerically stable implementation of Z=log(big_phi(beta)-big_phi(alpha))
	def _calculate_log_Z(self):
		# Values
		alpha_beta_sqr_diff = (self._alpha+self._beta)*(self._alpha-self._beta)
		beta_alpha_sqr_diff = -alpha_beta_sqr_diff
		out1 = -torch.log( (self._mean-self._mu) / self._sigma ) - LOG_SQRT_2_PI - (self._alpha**2)/2 + \
				torch.log(1 - torch.exp(alpha_beta_sqr_diff / 2))
		out2 = -torch.log( (self._mu-self._mean) / self._sigma ) - LOG_SQRT_2_PI - (self._beta**2)/2 + \
				torch.log(1 - torch.exp(beta_alpha_sqr_diff / 2))
		out3 = -LOG_2 + torch.log(erf(self._beta*INV_SQRT_2) - erf(self._alpha*INV_SQRT_2))

		# Conditions
		out1_cond = torch.logical_and(self._alpha>=0, self._beta>=0)
		out2_cond = torch.logical_and(self._alpha<=0, self._beta<=0)
		out3_cond = torch.logical_and(torch.logical_not(out1_cond), torch.logical_not(out2_cond))
		
		# Final expression
		# We need to set nan, -inf, inf values to 0.0 since, otherwise, 0*inf may result in NaNs
		# in the final formula
		final_out = torch.nan_to_num(out1,nan=0.0,posinf=0.0,neginf=0.0)*out1_cond + \
			  		torch.nan_to_num(out2,nan=0.0,posinf=0.0,neginf=0.0)*out2_cond + \
					torch.nan_to_num(out3,nan=0.0,posinf=0.0,neginf=0.0)*out3_cond

		return final_out

		"""
		Sequential code
		
		if self._alpha>=0 and self._beta>=0: # mu is smaller or equal than a and b #out1_cond
			alpha_beta_sqr_diff = (self._alpha+self._beta)*(self._alpha-self._beta) # alpha**2 - beta**2
			result = -torch.log( (self._mean-self._mu) / self._sigma ) - LOG_SQRT_2_PI - (self._alpha**2)/2 + \
					torch.log(1 - torch.exp(alpha_beta_sqr_diff / 2))
		
		elif self._alpha<=0 and self._beta<=0: # mu is larger or equal than a and b #out2_cond
			beta_alpha_sqr_diff = (self._beta+self._alpha)*(self._beta-self._alpha) # beta**2 - alpha**2
			result = -torch.log( (self._mu-self._mean) / self._sigma ) - LOG_SQRT_2_PI - (self._beta**2)/2 + \
					torch.log(1 - torch.exp(beta_alpha_sqr_diff / 2))
		
		else: # alpha and beta have different sign (which means mu lies in the interval [a,b]) #out3_cond
			result = -LOG_2 + torch.log(erf(self._beta*INV_SQRT_2) - erf(self._alpha*INV_SQRT_2))
		"""
		
	
	# --- Main methods ---

	# Mean of the distribution (after truncation)
	def _calculate_mean(self):
		cls = self.__class__
		fraction = (SQRT_2 / SQRT_PI) * cls._F_1(self._alpha*INV_SQRT_2, self._beta*INV_SQRT_2)
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