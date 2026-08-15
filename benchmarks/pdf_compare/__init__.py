"""Unified testing ground for the two multivariate alpha-stable density methods.

Two independent implementations estimate the joint density of a (symmetric or,
for the projection method, skewed) alpha-stable random vector from samples of
its spectral measure:

* **Projection method** -- ``aub_htp.pdf.multivariate.multivariate_alpha_stable_pdf``.
  Nolan/Matsui-Takemura one-dimensional-projection integral.  Point-wise, works
  for general skew and shift, but pays a quadrature cost per evaluation.
* **Inverse-FFT method** -- ``pdf_fft.py`` (repo root).  Monte-Carlo estimate of
  the (complex) characteristic function on a frequency grid followed by an
  inverse FFT.  One-shot on a grid; skewness enters through the imaginary part
  of the exponent and the location through an ``exp(i <t, mu>)`` phase.

This package wraps both behind a common :class:`DensityEstimator` interface and
provides references, error metrics, standard cases, and a benchmark harness so
the two can be compared on **correctness**, **speed of convergence** and
**performance**.

This is a standalone top-level package (not part of the installed ``aub_htp``
distribution).  See :mod:`benchmark` for the runnable harness and
``tests/multivariate_pdf_compare_test.py`` for the fast pytest guards.
"""

from .estimators import DensityEstimator, ProjectionEstimator, FFTEstimator
from .references import (
    spectral_samples,
    gaussian_covariance,
    gaussian_pdf,
    axis_marginal_params,
    axis_marginal_pdf,
    monte_carlo_marginal,
)
from .metrics import error_summary, integrate_grid, l1_distance, grid_marginal
from .cases import (
    Case,
    CROSS_METHOD_CASES,
    GAUSSIAN_CASES,
    SKEWED_CASES,
    PROJECTION_ONLY_CASES,
    ALL_CASES,
    get_case,
)

__all__ = [
    "DensityEstimator",
    "ProjectionEstimator",
    "FFTEstimator",
    "spectral_samples",
    "gaussian_covariance",
    "gaussian_pdf",
    "axis_marginal_params",
    "axis_marginal_pdf",
    "monte_carlo_marginal",
    "error_summary",
    "integrate_grid",
    "l1_distance",
    "grid_marginal",
    "Case",
    "CROSS_METHOD_CASES",
    "GAUSSIAN_CASES",
    "SKEWED_CASES",
    "PROJECTION_ONLY_CASES",
    "ALL_CASES",
    "get_case",
]
