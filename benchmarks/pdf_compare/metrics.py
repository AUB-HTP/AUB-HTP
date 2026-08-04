"""Error metrics shared by the pytest guards and the benchmark harness."""

from __future__ import annotations

import numpy as np


def error_summary(estimate, reference, *, rel_floor_frac=0.02):
    """Absolute and (bulk) relative error between two density arrays.

    Relative errors are reported only where the reference exceeds
    ``rel_floor_frac * max(|reference|)``, so the tiny-density tail (where
    relative error is meaningless and noise-dominated) does not swamp the
    summary.

    Returns
    -------
    dict with keys ``max_abs``, ``rmse``, ``mae``, ``median_rel``, ``max_rel``,
    ``n_bulk``.
    """
    e = np.asarray(estimate, dtype=np.float64).ravel()
    r = np.asarray(reference, dtype=np.float64).ravel()
    diff = e - r

    out = {
        "max_abs": float(np.max(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "mae": float(np.mean(np.abs(diff))),
    }

    thr = rel_floor_frac * float(np.max(np.abs(r))) if r.size else 0.0
    mask = np.abs(r) > thr
    if np.any(mask):
        rel = np.abs(diff[mask]) / np.abs(r[mask])
        out["median_rel"] = float(np.median(rel))
        out["max_rel"] = float(np.max(rel))
        out["n_bulk"] = int(mask.sum())
    else:
        out["median_rel"] = float("nan")
        out["max_rel"] = float("nan")
        out["n_bulk"] = 0
    return out


def integrate_grid(pdf_grid, dx, d):
    """Numerical integral of a density sampled on a regular grid."""
    return float(np.asarray(pdf_grid).sum() * dx ** d)


def l1_distance(estimate_grid, reference_grid, dx, d):
    """``L1`` distance ``integral |f - g|`` between two grid densities."""
    diff = np.asarray(estimate_grid) - np.asarray(reference_grid)
    return float(np.abs(diff).sum() * dx ** d)


def grid_marginal(pdf_grid, dx, axis=0):
    """Marginal of a 2D joint grid along ``axis`` (integrate the other axis).

    ``pdf_grid`` is in ``"ij"`` indexing (``pdf_grid[i, j] = f(x_i, y_j)``), so
    the marginal of coordinate ``axis`` sums over the *other* axis.
    """
    other = 1 - axis
    return np.asarray(pdf_grid).sum(axis=other) * dx
