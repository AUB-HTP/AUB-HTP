#!/usr/bin/env python
"""Cost of the kernel change: spline build and exact=True evaluation.

Uses the 30-degree rotation oracle measure (4 atoms) and a 5x5 grid of points.
Run from the repository root:  python benchmarks/diagnostics/kernel_timing.py
"""
import time
import warnings

import numpy as np

from aub_htp.pdf.multivariate import MultivariateStableDensity
from aub_htp.random.spectral_measure_sampler import DiscreteSampler

warnings.filterwarnings("ignore")
theta = np.pi / 6
A = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
positions = np.vstack([A.T, -A.T])
g = np.linspace(-6.0, 6.0, 5)
x = np.array([[a, b] for a in g for b in g]) @ A.T

MEASURES = {
    "symmetric": np.full(4, 0.5),
    "skewed": np.array([0.75, 0.35, 0.25, 0.65]),   # beta = (0.5, -0.3)
}

print(f"{'alpha':<6} {'measure':<10} {'spline build':>13} {'pdf (spline)':>13} {'pdf (exact)':>12}")
for alpha in (0.5, 0.8, 0.95):
    for name, weights in MEASURES.items():
        sampler = DiscreteSampler(alpha, positions, weights)
        t = time.perf_counter()
        model = MultivariateStableDensity(alpha, sampler)
        build = time.perf_counter() - t
        t = time.perf_counter()
        model.pdf(x)
        spline_eval = time.perf_counter() - t
        t = time.perf_counter()
        MultivariateStableDensity(alpha, sampler, exact=True).pdf(x)
        exact_eval = time.perf_counter() - t
        print(f"{alpha:<6} {name:<10} {build:>12.2f}s {spline_eval:>12.2f}s {exact_eval:>11.2f}s")