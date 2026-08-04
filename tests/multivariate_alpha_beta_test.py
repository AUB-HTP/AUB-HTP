"""Coverage of the whole ``alpha`` x ``beta`` parameter range for both methods.

A one-dimensional spectral measure with atoms at ``+1`` (weight ``wp``) and
``-1`` (weight ``wm``) generates a *univariate* S1 stable law with

    sigma^alpha = wp + wm            beta = (wp - wm) / (wp + wm)

so ``alpha_stable.pdf`` -- an independent implementation -- is an exact
reference for **every** ``beta`` in ``[-1, 1]``, which a symmetric multivariate
measure could never reach.  ``d == 1`` also makes the projection method's sphere
quadrature exact (``S^0 = {-1, +1}``), isolating the kernel.

Two properties of the S1 parameterization drive the test design:

* the location of the law moves like ``tan(pi alpha / 2)``, which diverges as
  ``alpha -> 1``, so the grid is centred on the reference's own mode instead of
  the origin (otherwise the bulk leaves the FFT window near ``alpha = 1``);
* for the same reason the *shared* Monte-Carlo error in the skewness estimate is
  amplified by ``tan(pi alpha / 2)``.  The sampler below is therefore
  antithetic: it returns exactly ``wp : wm`` signs, so the empirical skewness is
  exact and the comparison measures the density methods rather than the sampling
  noise.  See ``test_near_alpha1_skew_noise_is_amplified`` for the effect that
  choice suppresses.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aub_htp import alpha_stable  # noqa: E402
from aub_htp.pdf.multivariate import MultivariateStableDensity  # noqa: E402
from aub_htp.random.spectral_measure_sampler import (  # noqa: E402
    BaseSpectralMeasureSampler,
)
from aub_htp.random.util import get_random_state_generator  # noqa: E402
from benchmarks.pdf_compare import FFTEstimator  # noqa: E402


class _Atoms1D(BaseSpectralMeasureSampler):
    """1-D spectral measure on ``{-1, +1}`` realising an exact ``beta``."""

    def __init__(self, beta, alpha, sigma=1.0, antithetic=True):
        total = sigma ** alpha
        self.wp = 0.5 * total * (1.0 + beta)
        self.wm = 0.5 * total * (1.0 - beta)
        self._mass = self.wp + self.wm
        self._p_plus = self.wp / self._mass
        self._antithetic = antithetic

    def sample(self, number_of_samples, random_state=None):
        if self._antithetic:
            # exact wp : wm split -> empirical skewness equals the true one
            n_plus = int(round(self._p_plus * number_of_samples))
            signs = np.concatenate([np.ones(n_plus),
                                    -np.ones(number_of_samples - n_plus)])
        else:
            rng = get_random_state_generator(random_state)
            signs = np.where(rng.random(number_of_samples) <= self._p_plus,
                             1.0, -1.0)
        return signs.reshape(-1, 1)

    def dimensions(self):
        return 1

    def mass(self):
        return float(self._mass)


def _bulk_window(alpha, beta):
    """Mode and half-width of the reference density's bulk."""
    span = 40.0 if alpha > 0.9 else 80.0
    x = np.linspace(-span, span, 20_001)
    f = np.nan_to_num(alpha_stable.pdf(x, alpha, beta, loc=0.0, scale=1.0))
    mode = float(x[np.argmax(f)])
    kept = x[f > 0.02 * f.max()]
    return mode, max(float(np.max(np.abs(kept - mode))), 2.0)


def _compare(method, alpha, beta, n_spectral=4000):
    sampler = _Atoms1D(beta, alpha)
    mode, half = _bulk_window(alpha, beta)

    xs = np.linspace(-half, half, 301)
    ref = np.nan_to_num(alpha_stable.pdf(xs + mode, alpha, beta,
                                         loc=0.0, scale=1.0))
    keep = ref > 0.02 * ref.max()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "fft":
            extent = max(6.0 * half, 40.0)
            dx = min(half / 60.0, 0.05)
            grid = int(2 * round(extent / dx)) + 1
            est = FFTEstimator(number_of_spectral_samples=n_spectral,
                               grid_size=grid, dt=np.pi / extent)
            est.setup(alpha, sampler, shift=-mode, random_state=0)
            got = est.evaluate(xs.reshape(-1, 1))
        else:
            model = MultivariateStableDensity(
                alpha, sampler, shift=-mode,
                number_of_spectral_samples=n_spectral, random_state=0)
            got = model.pdf(xs.reshape(-1, 1))

    return float(np.median(np.abs(got - ref)[keep] / ref[keep]))


ALPHAS = [0.4, 0.6, 0.8, 0.95, 1.0, 1.05, 1.2, 1.5, 1.8, 1.95]
BETAS = [0.0, 0.5, 1.0, -1.0]


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("beta", BETAS)
@pytest.mark.parametrize("method", ["projection", "fft"])
def test_matches_univariate_reference(method, alpha, beta):
    """Both methods reproduce the exact 1-D law for all alpha and beta."""
    # The FFT reconstructs a whole grid, so small alpha (sharp peak plus heavy
    # tail) is resolution-limited where the point-wise method is not.
    tolerance = 0.03
    if method == "fft" and alpha < 0.7:
        tolerance = 0.08
    assert _compare(method, alpha, beta) < tolerance


@pytest.mark.parametrize("beta", [1.0, -1.0])
def test_fully_skewed_is_one_sided_for_alpha_below_one(beta):
    """For ``alpha < 1`` and ``beta = +-1`` the support is a half-line."""
    alpha = 0.6
    sampler = _Atoms1D(beta, alpha)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MultivariateStableDensity(alpha, sampler,
                                          number_of_spectral_samples=4000,
                                          random_state=0)
        # far on the forbidden side the density must be negligible
        forbidden = np.array([[-40.0 * beta]])
        allowed = np.array([[+3.0 * beta]])
        assert model.pdf(forbidden)[0] < 1e-4 * model.pdf(allowed)[0]


def test_near_alpha1_skew_noise_is_amplified():
    """Document the shared failure mode near ``alpha = 1``.

    The S1 characteristic function multiplies the skewness by
    ``tan(pi alpha / 2)``, which is ~12.7 at ``alpha = 0.95``.  A symmetric
    measure sampled i.i.d. has an empirical skewness of order ``1/sqrt(n)``, so
    the *spurious* phase is ``O(tan(pi alpha / 2) / sqrt(n))`` -- both methods
    inherit it identically, and it shrinks only like ``1 / sqrt(n)``.  An
    antithetic sampler removes it outright.
    """
    alpha = 0.95
    xs = np.linspace(-4.0, 4.0, 81)
    ref = np.nan_to_num(alpha_stable.pdf(xs, alpha, 0.0, loc=0.0, scale=1.0))
    keep = ref > 0.02 * ref.max()

    def error(antithetic):
        sampler = _Atoms1D(0.0, alpha, antithetic=antithetic)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = MultivariateStableDensity(
                alpha, sampler, number_of_spectral_samples=2000, random_state=0)
            got = model.pdf(xs.reshape(-1, 1))
        return float(np.median(np.abs(got - ref)[keep] / ref[keep]))

    exact_skew = error(antithetic=True)
    noisy_skew = error(antithetic=False)
    assert exact_skew < 0.01
    assert noisy_skew > 5.0 * max(exact_skew, 1e-4)


def test_projection_refuses_the_gaussian_limit():
    """``alpha == 2`` makes ``tan(pi alpha / 2)`` diverge; fail loudly."""
    from aub_htp import IsotropicSampler
    with pytest.raises(ValueError, match="alpha == 2"):
        MultivariateStableDensity(2.0, IsotropicSampler(2, 2.0, 1.0))
