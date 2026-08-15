"""Accuracy of the projection kernel ``g_{alpha,d}`` against its definition.

``g_{alpha,d}`` is evaluated by the module through three different
representations (the Matsui-Takemura finite-interval kernel for ``alpha != 1``,
direct panel quadrature, and a large-``|v|`` series for ``alpha == 1``), so it is
worth checking all of them against the defining integral

    g_{a,d}(v, b) = (2 pi)^{-d} int_0^inf cos(v u - b tan(pi a / 2) u^a)
                                          u^{d-1} exp(-u^a) du       (a != 1)
    g_{1,d}(v, b) = (2 pi)^{-d} int_0^inf cos(v u + (2/pi) b u log u)
                                          u^{d-1} exp(-u) du         (a == 1)

The reference integrand oscillates like ``cos(v u)`` over a range that grows
like ``39^(1/alpha)``, so it is integrated panel-by-panel; a single adaptive
call silently loses the phase and is wrong by orders of magnitude for large
``|v|``.

These cases are regression tests for three bugs that a coarser sweep missed:

* ``alpha == 1`` with ``beta != 0`` was wrong by ``pi^(d-1)`` and collapsed to
  zero for small ``|beta|`` (Zolotarev's (B) form carries ``exp(-x/beta)``);
* ``|beta| == 1`` hit an endpoint singularity of the finite-interval kernel;
* a spurious "extra term" doubled ``g`` at ``beta == 1`` for even ``d``.

The first two only show up for ``beta`` values away from ``{0, 0.5}`` and the
third only for even ``d``, which is why the sweep below covers ``beta`` up to
``+-1`` and both parities of ``d``.
"""

import numpy as np
import pytest
from scipy.integrate import quad

from aub_htp.pdf.multivariate import g_alpha_d

TWO_PI = 2.0 * np.pi


def _reference(v, beta, alpha, d):
    """Definition (1) by composite quadrature that resolves the oscillation."""
    if alpha == 1.0:
        def f(u):
            return (np.exp(-u) * u ** (d - 1)
                    * np.cos(v * u + (2.0 / np.pi) * beta * u * np.log(u)))
        upper = 45.0
    else:
        tau = np.tan(np.pi * alpha / 2.0)
        def f(u):
            return (np.cos(v * u - beta * tau * u ** alpha) * u ** (d - 1)
                    * np.exp(-u ** alpha))
        upper = float(min(39.0 ** (1.0 / alpha), 2.0e3))

    step = min(max(TWO_PI / max(abs(v), 1e-3), upper / 20_000.0), upper)
    edges = list(np.arange(0.0, upper, step)) + [upper]
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        value, _ = quad(f, lo, hi, limit=600, epsabs=1e-13, epsrel=1e-13)
        total += value
    return total / TWO_PI ** d


V_POINTS = np.array([0.0, 0.3, 1.0, 2.5, -0.7, -3.0, 8.0])


@pytest.mark.parametrize("alpha", [0.5, 0.8, 1.0, 1.2, 1.5, 1.9])
@pytest.mark.parametrize("beta", [0.0, 0.001, 0.4, 1.0, -1.0])
@pytest.mark.parametrize("d", [1, 2, 3])
def test_kernel_matches_definition(alpha, beta, d):
    got = g_alpha_d(V_POINTS, np.full_like(V_POINTS, beta), alpha, d)
    want = np.array([_reference(v, beta, alpha, d) for v in V_POINTS])

    peak = np.max(np.abs(want))
    # Compare on the scale of the kernel's own peak: g crosses zero, so a
    # pointwise relative tolerance is meaningless near the crossings.
    assert np.max(np.abs(got - want)) < 1e-5 * peak


@pytest.mark.parametrize("d", [1, 2, 3, 4])
def test_alpha1_symmetric_closed_form(d):
    """``beta == 0`` at ``alpha == 1`` has a closed form; check the general path."""
    import math
    v = np.array([0.0, 0.5, 2.0, -4.0, 30.0, -200.0])
    closed = (math.factorial(d - 1) / TWO_PI ** d
              * (1.0 + v ** 2) ** (-d / 2.0) * np.cos(d * np.arctan(v)))
    # a hair off zero, so the generic (panel / series) branch runs rather than
    # the closed-form shortcut
    got = g_alpha_d(v, np.full_like(v, 1e-12), 1.0, d)
    assert np.allclose(got, closed, rtol=1e-6, atol=1e-14 * np.max(np.abs(closed)))


def test_alpha1_panel_and_series_branches_agree():
    """The two ``alpha == 1`` regimes must join smoothly at the switch point."""
    from aub_htp.pdf.multivariate import (_A1_ASYMPTOTIC_V, _g_alpha1_panels,
                                          _g_alpha1_asymptotic)
    v = np.array([_A1_ASYMPTOTIC_V]) * np.array([[1.0], [-1.0]])
    for beta in (0.3, 1.0, -1.0):
        for vv in v.ravel():
            a = _g_alpha1_panels(np.array([vv]), np.array([beta]), 2)[0]
            b = _g_alpha1_asymptotic(np.array([vv]), np.array([beta]), 2)[0]
            assert abs(a - b) < 1e-5 * max(abs(a), 1e-12), (beta, vv, a, b)


def test_kernel_is_even_under_joint_sign_flip():
    """``g(v, beta) == g(-v, -beta)`` for every representation."""
    v = np.array([0.2, 1.3, 4.0, 25.0, 60.0])
    for alpha in (0.7, 1.0, 1.6):
        for beta in (0.0, 0.5, 1.0):
            a = g_alpha_d(v, np.full_like(v, beta), alpha, 2)
            b = g_alpha_d(-v, np.full_like(v, -beta), alpha, 2)
            peak = max(np.max(np.abs(a)), 1e-300)
            assert np.max(np.abs(a - b)) < 1e-8 * peak, (alpha, beta)
