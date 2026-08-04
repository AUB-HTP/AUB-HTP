"""Fast correctness guards for the two multivariate alpha-stable PDF methods.

These are the pass/fail counterpart to the ``pdf_compare`` benchmark harness
(which produces the quantitative convergence/performance study).  They exercise
both the projection method (``aub_htp.pdf.multivariate``) and the inverse-FFT
method (``pdf_fft.py``) through the shared ``pdf_compare`` interface and check
them against independent references:

* exact Gaussian limit at ``alpha == 2`` (FFT),
* the package's own 1D ``alpha_stable`` marginal,
* normalization (integral == 1) and non-negativity,
* radial symmetry of isotropic laws,
* cross-method agreement.

All checks run by default: the projection method now tabulates its kernel once
and evaluates points in bulk, so a full 2-D grid from it costs seconds rather
than the tens of minutes it used to.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# The comparison library is a top-level (non-installed) package at the repo
# root; add the root to the path so "import pdf_compare" resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.pdf_compare import FFTEstimator, ProjectionEstimator, cases as cases_mod, metrics
from benchmarks.pdf_compare import (  # noqa: E402
    references,
)

SEED = 0
N_SPECTRAL = 12_000
FFT_KW = dict(grid_size=81, dt=0.35)


def _anchor_points(d):
    e0 = np.zeros(d); e0[0] = 1.0
    diag = np.ones(d) / np.sqrt(d)
    return np.vstack([np.zeros(d), e0, e0 * 2.0, diag * 1.5, diag * 3.0, -e0 * 2.0])


# --------------------------------------------------------------------------- #
# Metric unit tests (pure, instant)
# --------------------------------------------------------------------------- #
def test_error_summary_zero_on_identical():
    a = np.array([0.1, 0.5, 1.0, 2.0])
    out = metrics.error_summary(a, a)
    assert out["max_abs"] == 0.0
    assert out["rmse"] == 0.0
    assert out["median_rel"] == 0.0


def test_integrate_grid_of_constant():
    # constant density 0.25 on a 4-unit square (dx=0.5, 5x5 cells sum) -> ~area
    grid = np.full((5, 5), 0.25)
    assert metrics.integrate_grid(grid, dx=0.5, d=2) == pytest.approx(0.25 * 25 * 0.25)


def test_grid_marginal_matches_manual():
    grid = np.arange(12, dtype=float).reshape(3, 4)
    marg0 = metrics.grid_marginal(grid, dx=2.0, axis=0)
    assert np.allclose(marg0, grid.sum(axis=1) * 2.0)


# --------------------------------------------------------------------------- #
# Interface / shape
# --------------------------------------------------------------------------- #
def test_estimator_output_shapes():
    case = cases_mod.get_case("iso2d_a1.5")
    sampler = case.sampler()
    fft = FFTEstimator(number_of_spectral_samples=N_SPECTRAL, **FFT_KW)
    fft.setup(case.alpha, sampler, random_state=SEED)
    pts = _anchor_points(2)
    vals = fft.evaluate(pts)
    assert vals.shape == (pts.shape[0],)
    grid = fft.joint_on_grid([np.linspace(-2, 2, 7)] * 2)
    assert grid.shape == (7, 7)


# --------------------------------------------------------------------------- #
# FFT: exact Gaussian limit at alpha == 2
# --------------------------------------------------------------------------- #
def test_fft_matches_gaussian_at_alpha_2():
    case = cases_mod.get_case("gauss2d")
    sampler = case.sampler()
    samples = references.spectral_samples(sampler, N_SPECTRAL, SEED)
    cov = references.gaussian_covariance(sampler, samples)

    fft = FFTEstimator(number_of_spectral_samples=N_SPECTRAL, **FFT_KW)
    fft.setup(case.alpha, sampler, random_state=SEED)

    pts = _anchor_points(2)
    got = fft.evaluate(pts)
    ref = references.gaussian_pdf(pts, cov)
    summary = metrics.error_summary(got, ref)
    assert summary["median_rel"] < 0.03
    assert summary["max_rel"] < 0.08


# --------------------------------------------------------------------------- #
# FFT: normalization, non-negativity, radial symmetry
# --------------------------------------------------------------------------- #
def test_fft_normalizes_to_one():
    case = cases_mod.get_case("iso2d_a1.5")
    fft = FFTEstimator(number_of_spectral_samples=N_SPECTRAL, **FFT_KW)
    fft.setup(case.alpha, case.sampler(), random_state=SEED)
    assert fft.normalization() == pytest.approx(1.0, abs=0.12)


def test_fft_negative_mass_is_small():
    case = cases_mod.get_case("iso2d_a1.5")
    fft = FFTEstimator(number_of_spectral_samples=N_SPECTRAL, **FFT_KW)
    fft.setup(case.alpha, case.sampler(), random_state=SEED)
    # FFT ringing may push a little mass negative; it should stay tiny.
    assert fft.negative_mass() < 0.02


def test_isotropic_density_is_radially_symmetric():
    case = cases_mod.get_case("iso2d_a1.5")
    fft = FFTEstimator(number_of_spectral_samples=N_SPECTRAL, **FFT_KW)
    fft.setup(case.alpha, case.sampler(), random_state=SEED)
    for radius in (1.0, 2.5):
        angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        pts = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
        vals = fft.evaluate(pts)
        cv = vals.std() / vals.mean()
        assert cv < 0.08, f"radius {radius}: CV={cv:.3f}"


# --------------------------------------------------------------------------- #
# Cross-method agreement (both methods, symmetric case)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case_name", ["iso2d_a1.2", "iso2d_a1.5", "iso2d_a1.8"])
def test_methods_agree_at_anchor_points(case_name):
    case = cases_mod.get_case(case_name)
    sampler = case.sampler()

    fft = FFTEstimator(number_of_spectral_samples=N_SPECTRAL, **FFT_KW)
    fft.setup(case.alpha, sampler, random_state=SEED)
    proj = ProjectionEstimator(number_of_spectral_samples=N_SPECTRAL)
    proj.setup(case.alpha, sampler, random_state=SEED)

    pts = _anchor_points(2)
    fv = fft.evaluate(pts)
    pv = proj.evaluate(pts)
    summary = metrics.error_summary(pv, fv)
    assert summary["median_rel"] < 0.15


# --------------------------------------------------------------------------- #
# Marginal agreement against the independent 1D implementation (needs a grid)
# --------------------------------------------------------------------------- #
def test_fft_marginal_matches_1d_stable():
    case = cases_mod.get_case("iso2d_a1.5")
    sampler = case.sampler()
    samples = references.spectral_samples(sampler, N_SPECTRAL, SEED)
    beta0, scale0 = references.axis_marginal_params(sampler, samples, case.alpha)

    fft = FFTEstimator(number_of_spectral_samples=N_SPECTRAL, **FFT_KW)
    fft.setup(case.alpha, sampler, random_state=SEED)
    marg = metrics.grid_marginal(fft.pdf_grid, fft.dx, axis=0)
    ref = references.axis_marginal_pdf(fft.axis, case.alpha, beta0, scale0)
    assert metrics.error_summary(marg, ref)["median_rel"] < 0.12


@pytest.mark.parametrize("method", ["projection", "fft"])
def test_skewed_marginal_matches_1d_stable(method):
    """Both methods must reproduce a genuinely skewed marginal.

    The skew lives on axis 1 (the y-marginal); axis 0 stays symmetric.  The FFT
    method carries it in the imaginary part of the characteristic function.
    """
    case = cases_mod.get_case("skew_star2d_a1.5")
    sampler = case.sampler()
    samples = references.spectral_samples(sampler, N_SPECTRAL, SEED)
    beta1, scale1 = references.axis_marginal_params(sampler, samples,
                                                    case.alpha, axis=1)
    assert abs(beta1) > 0.05  # the case must actually be skewed

    if method == "fft":
        est = FFTEstimator(number_of_spectral_samples=N_SPECTRAL, **FFT_KW)
        est.setup(case.alpha, sampler, random_state=SEED)
        marg = metrics.grid_marginal(est.pdf_grid, est.dx, axis=1)
        ref = references.axis_marginal_pdf(est.axis, case.alpha, beta1, scale1)
    else:
        est = ProjectionEstimator(number_of_spectral_samples=N_SPECTRAL)
        est.setup(case.alpha, sampler, random_state=SEED)
        axis = np.linspace(-10, 10, 61)
        grid = est.joint_on_grid([axis, axis])
        marg = metrics.grid_marginal(grid, axis[1] - axis[0], axis=1)
        ref = references.axis_marginal_pdf(axis, case.alpha, beta1, scale1)
    assert metrics.error_summary(marg, ref)["median_rel"] < 0.15


def test_skew_is_not_mirrored():
    """Guard the FFT inversion orientation.

    ``ifftn`` uses the +i convention, so it reconstructs f(-x); the estimator
    feeds the conjugate characteristic function to compensate.  Without that,
    skewed and shifted densities come out mirrored -- invisible for symmetric
    laws, which is how it went unnoticed.
    """
    case = cases_mod.get_case("shifted_iso2d_a1.5")
    sampler = case.sampler()
    fft = FFTEstimator(number_of_spectral_samples=N_SPECTRAL, **FFT_KW)
    fft.setup(case.alpha, sampler, shift=case.shift, random_state=SEED)

    # The mode must sit at +shift, not -shift.
    at_shift = fft.evaluate(np.asarray(case.shift, dtype=float)[None, :])[0]
    at_mirror = fft.evaluate(-np.asarray(case.shift, dtype=float)[None, :])[0]
    assert at_shift > 5.0 * at_mirror, (at_shift, at_mirror)


def test_projection_normalizes_to_one():
    case = cases_mod.get_case("iso2d_a1.5")
    proj = ProjectionEstimator(number_of_spectral_samples=N_SPECTRAL)
    proj.setup(case.alpha, case.sampler(), random_state=SEED)
    axis = np.linspace(-9, 9, 41)
    grid = proj.joint_on_grid([axis, axis])
    dx = axis[1] - axis[0]
    assert metrics.integrate_grid(grid, dx, d=2) == pytest.approx(1.0, abs=0.15)
