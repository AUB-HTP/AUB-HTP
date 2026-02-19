# aub_htp/__init__.py
"""
AUB Heavy-Tails Package (AUB-HTP)

A toolkit for working with heavy-tailed distributions,
including PDF generation and Random variable sampling.
"""
from .pdf import generate_alpha_stable_pdf
from .random import sample_alpha_stable_vector, BaseSpectralMeasureSampler

from ._alpha_stable import alpha_stable_gen, multivariate_alpha_stable_gen

alpha_stable = alpha_stable_gen("alpha_stable")
multivariate_alpha_stable = multivariate_alpha_stable_gen()