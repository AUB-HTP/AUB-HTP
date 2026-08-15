"""Common adapter interface over the two density implementations.

Both methods are driven from the *same* spectral-measure sampler and (via the
random seed) statistically identical spectral samples, so any difference the
harness measures is a genuine method difference rather than a difference of
inputs.

Both implementations have a build phase and an evaluation phase, but they
balance them very differently:

* the projection method is *point-wise* -- its build tabulates the projection
  parameters and the kernel, after which each point costs one sphere sum
  (``O(sphere nodes)`` per point, so :meth:`evaluate` scales linearly);
* the FFT method (``pdf_fft.py``) is *grid based* -- it pays a one-off cost to
  estimate the characteristic function and invert it on a whole grid, after
  which evaluation at arbitrary points is a cheap interpolation and essentially
  independent of the number of query points.

:class:`DensityEstimator` captures both patterns with a ``setup`` step (timed
into :attr:`setup_time`) and an ``evaluate`` step (timed into
:attr:`eval_time`), so the benchmark can attribute cost fairly.
"""

from __future__ import annotations

import contextlib
import io
import time
from abc import ABC, abstractmethod

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from aub_htp.pdf import pdf_fft as _fft
from aub_htp.pdf.multivariate import MultivariateStableDensity
from aub_htp.random.spectral_measure_sampler import BaseSpectralMeasureSampler


@contextlib.contextmanager
def _suppress_stdout():
    """Silence the (very chatty) debug prints in both implementations."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


class DensityEstimator(ABC):
    """Common interface: ``setup`` the model, then ``evaluate`` at points.

    Attributes
    ----------
    name : str
        Human-readable label used in reports and plots.
    setup_time, eval_time : float
        Wall-clock seconds spent in the most recent ``setup`` / ``evaluate``.
    """

    name = "base"

    def __init__(self):
        self.setup_time = 0.0
        self.eval_time = 0.0

    @abstractmethod
    def setup(self, alpha, sampler: BaseSpectralMeasureSampler, *,
              shift=0.0, random_state=None):
        """Prepare the estimator for a given law.  Returns ``self``."""

    @abstractmethod
    def evaluate(self, points) -> np.ndarray:
        """Density at ``points`` of shape ``(n, d)``.  Returns shape ``(n,)``."""

    def joint_on_grid(self, axes):
        """Evaluate the joint density on the tensor grid spanned by ``axes``.

        Parameters
        ----------
        axes : sequence of 1D arrays
            One coordinate axis per dimension.

        Returns
        -------
        grid : np.ndarray
            Density with shape ``tuple(len(a) for a in axes)`` in ``"ij"``
            (matrix) indexing, matching ``np.meshgrid(..., indexing="ij")``.
        """
        mesh = np.meshgrid(*axes, indexing="ij")
        points = np.stack([m.ravel() for m in mesh], axis=1)
        values = self.evaluate(points)
        return values.reshape(mesh[0].shape)


class ProjectionEstimator(DensityEstimator):
    """Point-wise Nolan/Matsui-Takemura projection density.

    ``setup`` builds a :class:`MultivariateStableDensity`, which performs the
    spectral sampling, sphere quadrature and kernel tabulation once; ``evaluate``
    then only pays for the kernel lookup and the sphere sum.

    Parameters
    ----------
    number_of_spectral_samples : int
        Monte-Carlo samples used to estimate ``sigma``/``beta`` per direction.
    number_of_sphere_points : int or None
        Sphere-quadrature nodes for the outer integral (``None`` -> method
        default: 360 for ``d == 2``, 4000 otherwise).
    exact : bool
        Bypass the kernel spline and evaluate the kernel directly (much slower;
        used as an accuracy reference).
    """

    name = "projection"

    def __init__(self, number_of_spectral_samples=20_000,
                 number_of_sphere_points=None, exact=False):
        super().__init__()
        self.number_of_spectral_samples = number_of_spectral_samples
        self.number_of_sphere_points = number_of_sphere_points
        self.exact = exact

    def setup(self, alpha, sampler, *, shift=0.0, random_state=None):
        start = time.perf_counter()
        with _suppress_stdout():
            self._model = MultivariateStableDensity(
                alpha,
                sampler,
                shift=shift,
                number_of_spectral_samples=self.number_of_spectral_samples,
                number_of_sphere_points=self.number_of_sphere_points,
                random_state=random_state,
                exact=self.exact,
            )
        self.setup_time = time.perf_counter() - start
        return self

    def evaluate(self, points):
        points = np.asarray(points, dtype=np.float64)
        start = time.perf_counter()
        with _suppress_stdout():
            values = self._model.pdf(points)
        self.eval_time = time.perf_counter() - start
        return np.atleast_1d(np.asarray(values, dtype=np.float64))


class FFTEstimator(DensityEstimator):
    """Grid-based inverse-FFT density (general: symmetric, skewed, or shifted).

    Builds the grid, estimates the (complex) characteristic function (batched
    over grid points to bound memory), inverts it, and exposes a linear
    interpolator for point-wise queries.  Skewness enters through the imaginary
    part of the CF and a location ``shift`` through the ``i<t,mu>`` phase, so the
    reconstructed grid is the full (possibly non-symmetric) density.

    Parameters
    ----------
    number_of_spectral_samples : int
        Monte-Carlo samples for the CF estimate.
    grid_size : int
        Number of grid points per axis; forced odd (the FFT grid must contain
        zero exactly).
    dt : float
        Frequency-grid spacing.  Spatial extent is ~``pi / dt`` and spatial
        resolution is ``dx = 2*pi / (grid_size * dt)``.
    max_matrix_elements : int
        Cap on ``batch * number_of_spectral_samples`` when forming the
        projection matrix, to keep the CF estimate within memory.
    """

    name = "fft"

    def __init__(self, number_of_spectral_samples=20_000, grid_size=129,
                 dt=0.25, max_matrix_elements=20_000_000):
        super().__init__()
        self.number_of_spectral_samples = number_of_spectral_samples
        self.grid_size = grid_size if grid_size % 2 == 1 else grid_size + 1
        self.dt = dt
        self.max_matrix_elements = max_matrix_elements

    def _estimate_cf(self, alpha, t_grid, samples, mass, shift):
        """Batched call to the real ``pdf_fft.estimate_cf`` (avoids OOM)."""
        d = samples.shape[1]
        flat = t_grid.reshape(-1, d)
        total = flat.shape[0]
        n_samples = samples.shape[0]
        batch = max(1, int(self.max_matrix_elements // max(n_samples, 1)))
        out = np.empty(total, dtype=np.complex128)
        for lo in range(0, total, batch):
            chunk = flat[lo:lo + batch]
            out[lo:lo + batch] = _fft.estimate_cf(alpha, chunk, samples, mass,
                                                  shift=shift)
        return out.reshape(t_grid.shape[:-1])

    def setup(self, alpha, sampler, *, shift=0.0, random_state=None):
        d = sampler.dimensions()
        self._d = d
        self._shift = np.broadcast_to(np.asarray(shift, dtype=np.float64), (d,))

        start = time.perf_counter()
        samples = sampler.sample(self.number_of_spectral_samples, random_state)
        mass = sampler.mass()
        t_grid, _ = _fft.make_frequency_grid(self.grid_size, self.dt, d)
        phi = self._estimate_cf(alpha, t_grid, samples, mass, self._shift)
        with _suppress_stdout():
            pdf_grid, dx = _fft.inverse_fourier_pdf(phi, self.dt)
        self.setup_time = time.perf_counter() - start

        self.pdf_grid = pdf_grid
        self.dx = dx
        self.axis = (np.arange(self.grid_size) - self.grid_size // 2) * dx
        self._interp = RegularGridInterpolator(
            tuple([self.axis] * d), pdf_grid,
            method="linear", bounds_error=False, fill_value=0.0,
        )
        return self

    def evaluate(self, points):
        # The shift is baked into the CF phase, so the grid already carries it.
        points = np.asarray(points, dtype=np.float64)
        start = time.perf_counter()
        values = self._interp(points)
        self.eval_time = time.perf_counter() - start
        return np.asarray(values, dtype=np.float64)

    # ---- native-grid diagnostics (no interpolation error) ---------------- #
    def normalization(self):
        """Numerical integral of the density over the whole grid."""
        return float(self.pdf_grid.sum() * self.dx ** self._d)

    def negative_mass(self):
        """Total |negative density| times cell volume (FFT ringing)."""
        neg = self.pdf_grid[self.pdf_grid < 0.0]
        return float(np.abs(neg).sum() * self.dx ** self._d)
