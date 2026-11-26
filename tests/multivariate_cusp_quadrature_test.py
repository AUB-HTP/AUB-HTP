"""Cusp-split quadrature on the circle for 2D discrete spectral measures."""
from math import gamma, pi

import numpy as np
import pytest
from scipy.integrate import quad

from aub_htp.pdf.multivariate import (
    MultivariateStableDensity,
    _cusp_breakpoints,
    _cusp_split_directions,
)
from aub_htp.random.spectral_measure_sampler import DiscreteSampler, IsotropicSampler

THETA = pi / 6
A = np.array([[np.cos(THETA), -np.sin(THETA)], [np.sin(THETA), np.cos(THETA)]])
POSITIONS = np.vstack([A.T, -A.T])          # atoms at +-A e1, +-A e2
MEASURES = {
    "symmetric": (np.full(4, 0.5), (0.0, 0.0)),
    "skewed": (np.array([0.75, 0.35, 0.25, 0.65]), (0.5, -0.3)),
}


def stable_pdf_1d(y, alpha, beta):
    """S1 parameterization, scale 1, location 0, alpha != 1 (Nolan 1997)."""
    if y < 0:
        return stable_pdf_1d(-y, alpha, -beta)
    tan = np.tan(pi * alpha / 2)
    theta0 = np.arctan(beta * tan) / alpha
    if y == 0:
        return (gamma(1 + 1 / alpha) * np.cos(theta0)
                / (pi * (1 + (beta * tan) ** 2) ** (1 / (2 * alpha))))
    a1 = alpha / (alpha - 1)

    def V(t):
        return (np.cos(alpha * theta0) ** (1 / (alpha - 1))
                * (np.cos(t) / np.sin(alpha * (theta0 + t))) ** a1
                * np.cos(alpha * theta0 + (alpha - 1) * t) / np.cos(t))

    c = y ** a1

    def f(t):
        h = c * V(t)
        return h * np.exp(-h) if np.isfinite(h) else 0.0

    lo, hi = -theta0, pi / 2
    if hi - lo <= 0:
        return 0.0
    ts = np.linspace(lo, hi, 2001)[1:-1]
    hs = np.array([c * V(t) for t in ts])
    peak = ts[int(np.nanargmin(np.abs(np.log(np.where(hs > 0, hs, np.nan)))))]
    val = sum(quad(f, a, b, epsabs=0, epsrel=1e-13, limit=500)[0]
              for a, b in ((lo, peak), (peak, hi)))
    return alpha * val / (pi * abs(alpha - 1) * y)


def reference(z, alpha, betas):
    return np.array([stable_pdf_1d(a, alpha, betas[0]) * stable_pdf_1d(b, alpha, betas[1])
                     for a, b in z])


def grid(n=5, r=6.0):
    g = np.linspace(-r, r, n)
    z = np.array([[a, b] for a in g for b in g])
    return z, z @ A.T


# --- the rule itself (fast) ------------------------------------------------ #
def test_breakpoints_are_perpendicular_to_atoms():
    breaks = np.degrees(_cusp_breakpoints(POSITIONS, np.full(4, 0.5)))
    np.testing.assert_allclose(breaks, [30.0, 120.0, 210.0, 300.0], atol=1e-12)


def test_breakpoints_skip_zero_weight_and_origin_atoms():
    positions = np.vstack([POSITIONS, [[0.0, 0.0], [1.0, 1.0]]])
    weights = np.array([0.5, 0.5, 0.5, 0.5, 1.0, 0.0])
    assert _cusp_breakpoints(positions, weights).size == 4


def test_rule_size_and_total_weight():
    points, q = _cusp_split_directions(POSITIONS, np.full(4, 0.5), 360)
    assert points.shape == (2144, 2)
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1.0, atol=1e-14)
    assert q.min() > 0
    assert q.sum() == pytest.approx(2 * pi, abs=1e-12)


@pytest.mark.parametrize("alpha", [0.5, 0.8, 1.5])
def test_cusp_integrand_is_integrated_accurately(alpha):
    """int |cos(theta - phi)|^alpha over the circle, phi off the 1-degree grid."""
    phi = 0.3
    exact = 2 * np.sqrt(pi) * gamma((alpha + 1) / 2) / gamma(alpha / 2 + 1)
    points, q = _cusp_split_directions([[np.cos(phi), np.sin(phi)]], [1.0], 360)
    theta = np.arctan2(points[:, 1], points[:, 0])
    got = q @ np.abs(np.cos(theta - phi)) ** alpha
    e = np.linspace(0, 2 * pi, 360, endpoint=False)
    equi = 2 * pi / 360 * np.sum(np.abs(np.cos(e - phi)) ** alpha)
    assert abs(got - exact) / exact < 1e-9
    assert abs(got - exact) < 1e-3 * abs(equi - exact)


# --- rule selection (fast) ------------------------------------------------- #
def test_auto_uses_cusp_rule_for_discrete_2d():
    sampler = DiscreteSampler(0.5, POSITIONS, np.full(4, 0.5))
    assert MultivariateStableDensity(0.5, sampler).directions.shape[0] == 2144
    equi = MultivariateStableDensity(0.5, sampler, circle_rule="equispaced")
    assert equi.directions.shape[0] == 360
    np.testing.assert_allclose(equi.quad_weights, 2 * pi / 360)


def test_auto_keeps_equispaced_for_other_samplers():
    model = MultivariateStableDensity(1.5, IsotropicSampler(2, 1.5, 1.0))
    assert model.directions.shape[0] == 360


def test_invalid_circle_rule_is_rejected():
    sampler = DiscreteSampler(0.5, POSITIONS, np.full(4, 0.5))
    with pytest.raises(ValueError):
        MultivariateStableDensity(0.5, sampler, circle_rule="gauss")
    with pytest.raises(ValueError):
        MultivariateStableDensity(1.5, IsotropicSampler(2, 1.5, 1.0), circle_rule="cusp")


# --- accuracy against the exact density (slow, exact kernel) -------------- #
@pytest.mark.slow
@pytest.mark.parametrize("alpha,measure", [(0.5, "symmetric"), (0.8, "symmetric"),
                                           (0.5, "skewed"), (0.8, "skewed")])
def test_density_matches_reference(alpha, measure):
    weights, betas = MEASURES[measure]
    z, x = grid()
    want = reference(z, alpha, betas)
    model = MultivariateStableDensity(alpha, DiscreteSampler(alpha, POSITIONS, weights),
                                      exact=True)
    got = model.pdf(x)
    bulk = want >= 0.01 * want.max()
    rel = np.abs(got - want) / want
    print(f"\nalpha={alpha} {measure}: bulk {rel[bulk].max():.1e}, all {rel.max():.1e}")
    assert rel[bulk].max() < 1e-7
    assert rel.max() < 2e-4


@pytest.mark.slow
def test_far_points():
    """z = (+-6, +-6), alpha = 0.5: points where the kernel peak is narrow."""
    alpha = 0.5
    z, x = grid()
    corners = [0, 4, 20, 24]
    want = reference(z[corners], alpha, (0.0, 0.0))
    model = MultivariateStableDensity(alpha, DiscreteSampler(alpha, POSITIONS, np.full(4, 0.5)),
                                      exact=True)
    np.testing.assert_allclose(model.pdf(x[corners]), want, rtol=1e-5)
