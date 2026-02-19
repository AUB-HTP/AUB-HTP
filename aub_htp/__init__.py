"""
AUB Heavy-Tails Package (AUB-HTP)

A toolkit for working with heavy-tailed distributions,
including PDF generation and Random variable sampling.
"""
from .pdf import *
from .random import *

from ._alpha_stable import alpha_stable_gen, multivariate_alpha_stable_gen

alpha_stable = alpha_stable_gen("alpha_stable")
multivariate_alpha_stable = multivariate_alpha_stable_gen()