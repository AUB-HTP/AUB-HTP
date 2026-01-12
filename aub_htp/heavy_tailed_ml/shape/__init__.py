# shape/__init__.py
"""
Shape estimation submodule for heavy-tailed PCA.

Provides three methods for estimating covariance-like "shape" matrices
from heavy-tailed data:

- Method 1: Ratio-based correlation estimation
- Method 2: Log-correlation method  
- Method 3: LLN demixing method
"""
from .api import estimate_shape
from .method1 import estimate_shape_method1
from .method2 import estimate_shape_method2
from .method3 import estimate_shape_method3
from .utilities import estimate_marginals_sigma, alpha_scale_factor

__all__ = [
    "estimate_shape",
    "estimate_shape_method1",
    "estimate_shape_method2", 
    "estimate_shape_method3",
    "estimate_marginals_sigma",
    "alpha_scale_factor",
]
