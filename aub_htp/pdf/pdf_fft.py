"""Inverse-FFT density estimation for multivariate alpha-stable laws.

This module provides reusable Monte-Carlo characteristic-function estimation
and inverse Fourier transform density reconstruction for multivariate
alpha-stable distributions.
"""

import numpy as np
from numpy.fft import fftshift, ifftshift, ifftn

from aub_htp import IsotropicSampler, sample_alpha_stable_vector


def sample_spectral_measure(n_samples, d, alpha):
    """Return mass, spectral-measure sampler, and iid spectral samples."""
    sampler = IsotropicSampler(d, alpha, 1)
    samples = sampler.sample(n_samples)
    return sampler.mass(), sampler, samples


def make_frequency_grid(n, dt, d):
    """Create a centered frequency grid containing zero exactly.

    Parameters
    ----------
    n : int
        Number of grid points per axis. Must be odd.
    dt : float
        Frequency spacing.
    d : int
        Number of dimensions.

    Returns
    -------
    grid : np.ndarray
        Frequency grid of shape ``(n, ..., n, d)``.
    freq : np.ndarray
        One-dimensional grid of frequencies.
    """
    assert n % 2 == 1

    freq = np.arange(-(n // 2), n // 2 + 1) * dt
    mesh = np.meshgrid(*([freq] * d), indexing="ij")
    grid = np.stack(mesh, axis=-1)
    return grid, freq


def estimate_cf(alpha, t_grid, S, mass, shift=0.0):
    """Monte-Carlo estimate of the complex characteristic function.

    Parameters
    ----------
    alpha : float
        Stability index.
    t_grid : np.ndarray
        Grid of frequency vectors of shape ``(..., d)``.
    S : np.ndarray
        Spectral samples of shape ``(n_spectral, d)``.
    mass : float
        Total spectral mass.
    shift : float or array_like, optional
        Location vector for the law.

    Returns
    -------
    phi : np.ndarray
        Complex characteristic function values on the frequency grid.
    """
    d = S.shape[1]
    t_flat = t_grid.reshape(-1, d)
    projections = t_flat @ S.T
    abs_proj = np.abs(projections)

    re = mass * np.mean(abs_proj ** alpha, axis=1)

    if alpha != 1.0:
        im = (np.tan(np.pi * alpha / 2.0) * mass
              * np.mean(np.sign(projections) * abs_proj ** alpha, axis=1))
    else:
        norms = np.linalg.norm(S, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_unit = np.log(abs_proj / norms[None, :])
        log_unit = np.nan_to_num(log_unit, nan=0.0, posinf=0.0, neginf=0.0)
        im = -(2.0 / np.pi) * mass * np.mean(projections * log_unit, axis=1)

    shift = np.asarray(shift, dtype=np.float64)
    if shift.ndim > 0 or shift != 0.0:
        im = im + t_flat @ np.broadcast_to(shift, (d,))

    phi = np.exp(-re - 1j * im)
    return phi.reshape(t_grid.shape[:-1])


def inverse_fourier_pdf(phi, dt, verbose=False):
    """Invert the characteristic function to a density grid.

    Parameters
    ----------
    phi : np.ndarray
        Complex characteristic function values on a centered grid.
    dt : float
        Frequency spacing.
    verbose : bool, default False
        Print diagnostics when True.

    Returns
    -------
    pdf : np.ndarray
        Reconstructed density grid.
    dx : float
        Spacing in the spatial domain.
    """
    d = phi.ndim
    n = phi.shape[0]
    dx = 2.0 * np.pi / (n * dt)

    pdf = fftshift(ifftn(ifftshift(phi)))
    pdf = np.real(pdf)
    pdf *= (n * dt / (2.0 * np.pi)) ** d

    if verbose:
        print("\n========== FFT diagnostics ===========")
        print("dimension:", d)
        print("grid:", phi.shape)
        print("dt =", dt)
        print("dx =", dx)
        print("\nPDF diagnostics")
        print("----------------")
        print("min pdf:", pdf.min())
        print("max pdf:", pdf.max())
        negative = np.sum(np.abs(pdf[pdf < 0]))
        print("negative mass:", negative)
        print("PDF integral:", pdf.sum() * dx**d)

    return pdf, dx
