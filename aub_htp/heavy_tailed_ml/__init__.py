# heavy_tailed_ml/__init__.py
"""
Heavy-Tailed Machine Learning module for AUB-HTP.

This module provides robust statistical and machine learning methods
for heavy-tailed data where classical methods (that assume Gaussian-like
distributions) fail.

Key Concepts
------------
- α (alpha): Stability parameter in (0, 2]. α=2 is Gaussian, α=1 is Cauchy.
- α-power: Robust scale measure that replaces variance for heavy-tailed data.
- α-location: Robust center estimate that replaces the mean.

Main Functions
--------------
P_alpha : Compute α-power (robust scale measure)
location_L : Compute α-location (robust center)
ht_linear_regression_fit_nd : Heavy-tailed linear regression
kmeans_heavy_tailed : Heavy-tailed k-means clustering
estimate_shape : Shape matrix estimation for PCA

Utilities
---------
load_entropy_lookup : Load pre-computed entropies for α ≠ 1
load_rho_lookup : Load ρ → f(ρ) mapping for Method 2

Examples
--------
>>> import numpy as np
>>> from aub_htp import P_alpha, location_L, load_entropy_lookup
>>> 
>>> # For α = 1 (Cauchy), no lookup needed
>>> data = np.random.standard_cauchy(1000)
>>> mu, P = location_L(data, alpha=1.0)
>>> 
>>> # For α ≠ 1, load the entropy lookup
>>> h_lookup = load_entropy_lookup()
>>> mu, P = location_L(data, alpha=1.5, h_lookup=h_lookup)
"""
import json
from pathlib import Path

# Core functions
from .power import P_alpha
from .location import location_L
from .regression import (
    ht_linear_regression_fit_nd,
    ht_predict_nd,
    HeavyTailedLinearRegressionND,
)
from .kmeans import kmeans_heavy_tailed
from .shape import estimate_shape

# Data directory
_DATA_DIR = Path(__file__).parent / "data"


def load_entropy_lookup() -> dict:
    """
    Load pre-computed entropy values for α-power computation.
    
    Required when computing α-power for α ≠ 1.
    
    Returns
    -------
    dict
        Mapping from α (float) to entropy h(α) (float).
    
    Examples
    --------
    >>> h_lookup = load_entropy_lookup()
    >>> P = P_alpha(residuals, alpha=1.5, h_lookup=h_lookup)
    """
    with open(_DATA_DIR / "lookup_table_entropy.json") as f:
        raw = json.load(f)
    return {float(k): float(v) for k, v in raw.items()}


def load_rho_lookup() -> dict:
    """
    Load ρ → f(ρ) mapping for Method 2 shape estimation.
    
    Required when using estimate_shape with method="method2".
    
    Returns
    -------
    dict
        Mapping from ρ (correlation) to f(ρ).
    
    Examples
    --------
    >>> rho_lookup = load_rho_lookup()
    >>> result = estimate_shape(X, method="method2", lookup_rho_to_f=rho_lookup)
    """
    with open(_DATA_DIR / "lookup_table_rho.json") as f:
        raw = json.load(f)
    return {float(k): float(v) for k, v in raw.items()}


__all__ = [
    # Core functions
    "P_alpha",
    "location_L",
    # Regression
    "ht_linear_regression_fit_nd",
    "ht_predict_nd",
    "HeavyTailedLinearRegressionND",
    # Clustering
    "kmeans_heavy_tailed",
    # Shape estimation
    "estimate_shape",
    # Utilities
    "load_entropy_lookup",
    "load_rho_lookup",
]
