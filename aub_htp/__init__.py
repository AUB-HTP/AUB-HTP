# aub_htp/__init__.py
"""
AUB Heavy-Tails Package (AUB-HTP)

A comprehensive toolkit for working with heavy-tailed distributions,
including PDF generation and machine learning methods.

Modules
-------
alpha_stable_pdf : Generate PDFs of alpha-stable distributions
heavy_tailed_ml : Machine learning methods for heavy-tailed data

PDF Generation
--------------
generate_alpha_stable_pdf : Compute PDF of alpha-stable distributions

Heavy-Tailed ML
---------------
P_alpha : α-power computation (robust scale measure)
location_L : α-location estimation (robust center)
ht_linear_regression_fit_nd : Heavy-tailed linear regression
kmeans_heavy_tailed : Heavy-tailed k-means clustering
estimate_shape : Shape matrix estimation for PCA
load_entropy_lookup : Load entropy lookup table
load_rho_lookup : Load rho lookup table for Method 2

Examples
--------
>>> import numpy as np
>>> from aub_htp import generate_alpha_stable_pdf
>>> 
>>> # Generate PDF for Cauchy distribution
>>> x = np.linspace(-10, 10, 1000)
>>> pdf = generate_alpha_stable_pdf(x, alpha=1.0, beta=0, gamma=1, delta=0)

>>> from aub_htp import location_L, P_alpha
>>> 
>>> # Robust location and scale for Cauchy data
>>> data = np.random.standard_cauchy(1000)
>>> mu, P = location_L(data, alpha=1.0)
"""
# PDF generation
from .alpha_stable_pdf import generate_alpha_stable_pdf

# Heavy-tailed ML
from .heavy_tailed_ml import (
    P_alpha,
    location_L,
    ht_linear_regression_fit_nd,
    ht_predict_nd,
    HeavyTailedLinearRegressionND,
    kmeans_heavy_tailed,
    estimate_shape,
    load_entropy_lookup,
    load_rho_lookup,
)

__all__ = [
    # PDF generation
    "generate_alpha_stable_pdf",
    # Heavy-tailed ML - Core
    "P_alpha",
    "location_L",
    # Heavy-tailed ML - Regression
    "ht_linear_regression_fit_nd",
    "ht_predict_nd",
    "HeavyTailedLinearRegressionND",
    # Heavy-tailed ML - Clustering
    "kmeans_heavy_tailed",
    # Heavy-tailed ML - Shape estimation
    "estimate_shape",
    # Heavy-tailed ML - Utilities
    "load_entropy_lookup",
    "load_rho_lookup",
]
