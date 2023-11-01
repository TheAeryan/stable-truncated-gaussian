# __init__.py

__version__ = "1.3.7"

from .parallel_trunc_gaussian import ParallelTruncatedGaussian as TruncatedGaussian
# SequentialTruncatedGaussian is legacy code at this point, and lacks many methods such as rsample and icdf
# so I don't recomend you use it
from .sequential_trunc_gaussian import SequentialTruncatedGaussian as SeqTruncatedGaussian
