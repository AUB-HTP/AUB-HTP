#!/usr/bin/env python
"""Exact kernel g_alpha_d against the defining integral (_g_general_direct).

single: each v evaluated in its own call
batch:  all v evaluated in one call
Error is max |g - direct| / max |direct| over v in [1e-6, 100].

Run from the repository root:  python benchmarks/diagnostics/kernel_check.py
"""
import warnings

import numpy as np

from aub_htp.pdf.multivariate import _g_general_direct, g_alpha_d

warnings.filterwarnings("ignore")
vs = np.logspace(-6, 2, 33)
failed = 0
print(f"{'alpha':<7} {'beta':>5}    {'single':<9} {'worst v':<10} {'batch':<9} result")
for alpha in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.2, 1.5):
    for beta in (0.0, 0.5, -0.9):
        b = np.full_like(vs, beta)
        ref = _g_general_direct(vs, b, alpha, 2)
        single = np.array([g_alpha_d(vs[i:i + 1], b[i:i + 1], alpha, 2)[0]
                           for i in range(vs.size)])
        batch = g_alpha_d(vs, b, alpha, 2)
        scale = np.abs(ref).max()
        e_single = np.abs(single - ref) / scale
        e_batch = np.abs(batch - ref) / scale
        bad = max(e_single.max(), e_batch.max()) > 1e-6
        failed += bad
        print(f"{alpha:<7} {beta:>5}    {e_single.max():.1e}   "
              f"{vs[e_single.argmax()]:<9.2g}  {e_batch.max():.1e}   "
              + ("FAIL" if bad else ""))
print(f"\n{failed} failing combinations")
