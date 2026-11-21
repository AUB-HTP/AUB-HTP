"""_g_general_direct against definition (1) evaluated with mpmath."""
import warnings
import mpmath as mp
import numpy as np
from aub_htp.pdf.multivariate import _g_general_direct

warnings.filterwarnings("ignore")
mp.mp.dps = 20


def reference(v, beta, alpha, d=2):
    t = mp.tan(mp.pi * alpha / 2)
    f = lambda u: mp.cos(v * u - beta * t * u ** alpha) * u ** (d - 1) * mp.exp(-u ** alpha)
    if v > 0.5:
        val = mp.quadosc(f, [0, mp.inf], omega=v)
    else:
        val = mp.quad(f, [0, 1, 10, 100, 1000, 1e4, mp.inf])
    return float(val / (2 * mp.pi) ** d)


CASES = [(0.3, 0.5, 1e-3), (0.5, 0.5, 1e-6), (0.5, 0.0, 1e-4), (0.6, -0.9, 1e-3),
         (0.7, 0.5, 1e-4), (0.8, 0.5, 3.0), (0.95, 0.5, 0.1), (0.99, 0.5, 1.0),
         (0.99, 0.5, 56.2), (0.99, -0.9, 300.0)]
for alpha, beta, v in CASES:
    got = _g_general_direct(np.array([v]), np.array([beta]), alpha, 2)[0]
    want = reference(v, beta, alpha)
    print(f"alpha={alpha:<5} beta={beta:<5} v={v:<8g} direct {got: .8e}  "
          f"mpmath {want: .8e}  rel {abs(got - want) / abs(want):.1e}")
