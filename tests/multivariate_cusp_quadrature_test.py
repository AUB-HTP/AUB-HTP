"""Cusp-split quadrature on the circle for 2D discrete spectral measures."""
from math import gamma, pi

import numpy as np
import pytest
from scipy.integrate import quad

from aub_htp.pdf.multivariate import (
    MultivariateStableDensity,
    _cusp_breakpoints,
    _cusp_split_directions,
    g_alpha_d,
)
from aub_htp.random.spectral_measure_sampler import DiscreteSampler, IsotropicSampler

THETA = pi / 6
A = np.array([[np.cos(THETA), -np.sin(THETA)], [np.sin(THETA), np.cos(THETA)]])
POSITIONS = np.vstack([A.T, -A.T])          # atoms at +-A e1, +-A e2
MEASURES = {
    "symmetric": (np.full(4, 0.5), (0.0, 0.0)),
    "skewed": (np.array([0.75, 0.35, 0.25, 0.65]), (0.5, -0.3)),
}


def _unit(deg):
    r = np.radians(deg)
    return np.array([np.cos(r), np.sin(r)])


# Two uneven atom lines: at 10 and 75 degrees, with scales 1 and 2.
B = np.column_stack([_unit(10.0), _unit(75.0)])
SCALES = np.array([1.0, 2.0])

# Three atoms 120 degrees apart. Their sum is exactly zero, so DiscreteSampler
# accepts them for alpha >= 1, and every projection is skewed.
TRIANGLE = np.array([[0.0, 1.0], [-np.sqrt(3) / 2, -0.5], [np.sqrt(3) / 2, -0.5]])


def stable_pdf_1d(y, alpha, beta):
    """S1 parameterization, scale 1, location 0 (Nolan 1997; Cauchy at alpha = 1)."""
    if alpha == 1.0:
        if beta != 0.0:
            raise ValueError("only the symmetric case is implemented at alpha = 1")
        return 1.0 / (pi * (1.0 + y * y))
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


def uneven_measure(alpha, betas):
    """X = B diag(SCALES) Y with independent standard Y_i ~ S1(alpha, beta_i)."""
    s = SCALES ** alpha
    b = np.asarray(betas, dtype=float)
    positions = np.vstack([B.T, -B.T])
    weights = np.concatenate([s * (1 + b) / 2, s * (1 - b) / 2])
    return positions, weights


def uneven_grid(n=5, r=6.0):
    g = np.linspace(-r, r, n)
    y = np.array([[a, b] for a in g for b in g])
    return y, (y * SCALES) @ B.T


def uneven_reference(y, alpha, betas):
    return reference(y, alpha, betas) / (SCALES.prod() * abs(np.linalg.det(B)))


# --- the rule itself (fast) ------------------------------------------------ #
def test_breakpoints_are_perpendicular_to_atoms():
    breaks = np.degrees(_cusp_breakpoints(POSITIONS, np.full(4, 0.5)))
    np.testing.assert_allclose(breaks, [30.0, 120.0], atol=1e-12)


def test_breakpoints_skip_zero_weight_and_origin_atoms():
    positions = np.vstack([POSITIONS, [[0.0, 0.0], [1.0, 1.0]]])
    weights = np.array([0.5, 0.5, 0.5, 0.5, 1.0, 0.0])
    assert _cusp_breakpoints(positions, weights).size == 2


def test_near_coincident_cusps_are_merged():
    a = 0.3
    near = np.array([_unit(np.degrees(a)), _unit(np.degrees(a + 1e-8))])
    far = np.array([_unit(np.degrees(a)), _unit(np.degrees(a + 1e-3))])
    assert _cusp_breakpoints(near, [1.0, 1.0]).size == 1
    assert _cusp_breakpoints(far, [1.0, 1.0]).size == 2


def test_merging_wraps_around():
    eps = 1e-8                         # cusps at eps and pi - eps (mod pi)
    positions = np.array([[np.cos(-pi / 2 + eps), np.sin(-pi / 2 + eps)],
                          [np.cos(pi / 2 - eps), np.sin(pi / 2 - eps)]])
    breaks = _cusp_breakpoints(positions, [1.0, 1.0])
    assert breaks.size == 1
    assert min(breaks[0], pi - breaks[0]) < 1e-7


def test_rule_size_and_total_weight():
    points, q = _cusp_split_directions(POSITIONS, np.full(4, 0.5), 360)
    assert points.shape == (1072, 2)
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1.0, atol=1e-14)
    assert q.min() > 0
    assert q.sum() == pytest.approx(2 * pi, abs=1e-12)


@pytest.mark.parametrize("lines,expected", [(1, 944), (2, 1072), (5, 1360),
                                            (10, 2000), (20, 2560), (50, 5200)])
def test_rule_size_for_evenly_spaced_lines(lines, expected):
    angles = pi * np.arange(lines) / lines
    positions = np.column_stack([np.cos(angles), np.sin(angles)])
    points, q = _cusp_split_directions(positions, np.ones(lines), 360)
    assert points.shape[0] == expected
    assert q.sum() == pytest.approx(2 * pi, abs=1e-10)


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


@pytest.mark.parametrize("alpha,positions,weights", [
    (0.5, POSITIONS, MEASURES["skewed"][0]),
    (0.8, POSITIONS, MEASURES["skewed"][0]),
    (1.0, TRIANGLE, np.ones(3)),
    (1.5, TRIANGLE, np.ones(3)),
])
def test_integrand_is_even_on_the_circle(alpha, positions, weights):
    """The half-circle rule relies on f(s) = f(-s)."""
    model = MultivariateStableDensity(alpha, DiscreteSampler(alpha, positions, weights),
                                      exact=True, circle_rule="equispaced",
                                      number_of_sphere_points=16)
    x = np.array([[0.3, -1.2], [2.0, 0.7], [-4.0, 5.0]])
    v = model._v_of(x @ model.directions.T)
    g = g_alpha_d(v.ravel(), np.broadcast_to(model.beta, v.shape).ravel(),
                  alpha, 2).reshape(v.shape)
    f = g * model._inv_sigma_d
    np.testing.assert_allclose(f[:, :8], f[:, 8:], rtol=1e-8, atol=1e-14)


# --- rule selection (fast) ------------------------------------------------- #
def test_auto_uses_cusp_rule_for_discrete_2d():
    sampler = DiscreteSampler(0.5, POSITIONS, np.full(4, 0.5))
    assert MultivariateStableDensity(0.5, sampler).directions.shape[0] == 1072
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


# --- accuracy against exact densities (slow, exact kernel) ----------------- #
def _errors(got, want):
    bulk = want >= 0.01 * want.max()
    rel = np.abs(got - want) / want
    return rel[bulk].max(), rel.max()


@pytest.mark.slow
@pytest.mark.parametrize("alpha,measure", [(0.5, "symmetric"), (0.8, "symmetric"),
                                           (0.5, "skewed"), (0.8, "skewed")])
def test_density_matches_reference(alpha, measure):
    weights, betas = MEASURES[measure]
    z, x = grid()
    want = reference(z, alpha, betas)
    got = MultivariateStableDensity(alpha, DiscreteSampler(alpha, POSITIONS, weights),
                                    exact=True).pdf(x)
    bulk, worst = _errors(got, want)
    print(f"\nalpha={alpha} {measure}: bulk {bulk:.1e}, all {worst:.1e}")
    assert bulk < 1e-7
    assert worst < 2e-4


@pytest.mark.slow
@pytest.mark.parametrize("alpha,betas", [(0.7, (0.4, -0.6)), (1.0, (0.0, 0.0)),
                                         (1.5, (0.0, 0.0))])
def test_uneven_atoms_match_reference(alpha, betas):
    positions, weights = uneven_measure(alpha, betas)
    y, x = uneven_grid()
    want = uneven_reference(y, alpha, betas)
    got = MultivariateStableDensity(alpha, DiscreteSampler(alpha, positions, weights),
                                    exact=True).pdf(x)
    bulk, worst = _errors(got, want)
    print(f"\nuneven alpha={alpha} betas={betas}: bulk {bulk:.1e}, all {worst:.1e}")
    assert bulk < 1e-6
    assert worst < 1e-3


@pytest.mark.slow
@pytest.mark.parametrize("alpha", [1.0, 1.5])
def test_three_lines_self_convergence(alpha):
    """No closed form here, so the default rule is compared with a 4x finer one."""
    sampler = DiscreteSampler(alpha, TRIANGLE, np.ones(3))
    g = np.linspace(-4.0, 4.0, 5)
    x = np.array([[a, b] for a in g for b in g])
    coarse = MultivariateStableDensity(alpha, sampler, exact=True).pdf(x)
    fine = MultivariateStableDensity(alpha, sampler, exact=True,
                                     number_of_sphere_points=1440).pdf(x)
    bulk, worst = _errors(coarse, fine)
    print(f"\ntriangle alpha={alpha}: bulk {bulk:.1e}, all {worst:.1e}")
    assert bulk < 1e-6


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