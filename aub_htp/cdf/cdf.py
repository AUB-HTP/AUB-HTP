import numpy as np
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
from functools import lru_cache

from ..pdf.pdf import normalize_inputs
from .zolotarev import generate_cdf as generate_cdf_zolotarev
from .numericalIntegrationOfWrapperPDF import generate_cdf as generate_cdf_numerical
from .skorohod import generate_cdf as generate_cdf_skorohod


@lru_cache(maxsize=None)
def load_interpolator(name: str) -> RegularGridInterpolator:
    """
    Load a RegularGridInterpolator from an npz file.

    The npz file should contain:
    - grid_0, grid_1: 1D arrays defining the interpolation grid axes
    - values: 2D array of values on the grid
    - method: interpolation method (e.g. 'nearest')
    - bounds_error: whether to raise error for out-of-bounds
    - fill_value: value for out-of-bounds points
    - fill_value_is_none: whether fill_value should be None
    """
    npz_path = Path(__file__).parent / "data" / name
    data = np.load(npz_path, allow_pickle=True)

    fill_val = None if data['fill_value_is_none'].item() else data['fill_value'].item()

    return RegularGridInterpolator(
        (data['grid_0'], data['grid_1']),
        data['values'],
        method=str(data['method']),
        bounds_error=data['bounds_error'].item(),
        fill_value=fill_val
    )


def generate_switching_intervals(alpha, beta):
    """
    Intervals over x >= 0 and the method chosen for each, per (alpha,beta).
    - Upper bounds and method codes are read from interpolators, as in the pdf.
    - Method codes are 0 Zolotarev, 1 numerical integration, 2 Skorohod series.
    - A code of -1 marks an unused slot, so the walk stops there.
    - The first interval opens at -inf so that x = 0 falls inside it.
    Returns a list of (lower_bound, upper_bound, method) tuples.
    """
    upper_bound_fn = load_interpolator("switching_interpolator_boundaries.npz")
    method_fn = load_interpolator("switching_interpolator_methods.npz")

    upper_bounds = upper_bound_fn([(alpha, beta)])[0]
    methods = method_fn([(alpha, beta)])[0]

    intervals = []
    lower_bound = -np.inf

    for upper_bound, method in zip(upper_bounds, methods):
        if method < 0:
            break
        intervals.append((lower_bound, float(upper_bound), int(round(method))))
        lower_bound = float(upper_bound)

    return intervals


def generate_cdf_positive_side(X, alpha, beta, clip=True, enforce_monotone=True):
    """
    Piecewise cdf on x >= 0.
    - Each tabulated interval is evaluated by the method selected for it:
      • Zolotarev integral form near the centre
      • numerical integration of the pdf wrapper through the body
      • Skorohod tail series in the tail
    - Any point its method could not produce is refilled from the others, so
      the result never contains nan.
    - Optionally clip into [0,1] and force the result to be non-decreasing.
      Repair happens here, not on the assembled two-sided curve: a cumulative
      maximum applied there would sweep in opposite directions relative to the
      reflection and would break the beta-reflection identity.
    """
    cdf = np.full_like(X, np.nan)

    for lower_bound, upper_bound, method in generate_switching_intervals(alpha, beta):
        mask = (X > lower_bound) & (X <= upper_bound)

        if not np.any(mask):
            continue

        if method == 0:
            cdf[mask] = generate_cdf_zolotarev(X[mask], alpha, beta)
        elif method == 1:
            cdf[mask] = generate_cdf_numerical(X[mask], alpha, beta)
        elif method == 2:
            cdf[mask] = generate_cdf_skorohod(X[mask], alpha, beta)

    # Refill, most broadly applicable method first. Zolotarev raises for
    # alpha = 1, beta = 0, so each attempt is guarded.
    for generate_cdf_method in (generate_cdf_numerical,
                                generate_cdf_zolotarev,
                                generate_cdf_skorohod):
        mask_missing = ~np.isfinite(cdf)

        if not np.any(mask_missing):
            break

        try:
            values = generate_cdf_method(X[mask_missing], alpha, beta)
        except Exception:
            continue

        cdf[mask_missing] = np.where(np.isfinite(values), values, cdf[mask_missing])

    if clip:
        cdf = np.clip(cdf, 0.0, 1.0)

    if enforce_monotone:
        order = np.argsort(X, kind="stable")
        cdf[order] = np.maximum.accumulate(cdf[order])

    return cdf


def alpha_stable_cdf_core(X, alpha, beta, clip=True, enforce_monotone=True):
    """
    Core cdf on the normalized grid (unit scale), for 0 < alpha <= 2.
    - Evaluate x > 0 directly from the tabulated intervals.
    - Handle x < 0 by reflection with beta -> -beta, using the exact identity
      F(x; alpha, beta) = 1 - F(-x; alpha, -beta), so that symmetry and
      beta-reflection hold exactly rather than approximately.
    - x = 0 is claimed by both halves, so average the two estimates. That
      forces the identity to hold at the origin and returns exactly 0.5 when
      beta = 0.
    """
    if not (0 < alpha <= 2):
        raise Exception("Invalid alpha value")

    if not (-1 <= beta <= 1):
        raise Exception("Invalid beta value")

    cdf = np.zeros_like(X)

    mask_pos = X > 0
    if np.any(mask_pos):
        cdf[mask_pos] = generate_cdf_positive_side(
            X[mask_pos], alpha, beta, clip, enforce_monotone
        )

    mask_neg = X < 0
    if np.any(mask_neg):
        cdf[mask_neg] = 1.0 - generate_cdf_positive_side(
            -X[mask_neg], alpha, -beta, clip, enforce_monotone
        )

    mask_zero = X == 0
    if np.any(mask_zero):
        origin = np.zeros(1, dtype=np.float64)
        from_right = generate_cdf_positive_side(
            origin, alpha, beta, clip, enforce_monotone
        )[0]
        from_left = generate_cdf_positive_side(
            origin, alpha, -beta, clip, enforce_monotone
        )[0]
        cdf[mask_zero] = 0.5 * (from_right + 1.0 - from_left)

    return cdf


def generate_alpha_stable_cdf(X, alpha, beta, gamma=1.0, delta=0.0,
                              clip=True, enforce_monotone=True):
    """
    Public cdf wrapper.
    Pipeline:
    1) Normalize the query grid to unit scale, sharing the pdf's convention.
    2) Evaluate the piecewise cdf by alpha regime and tabulated interval.
    3) Return F, with no 1/gamma factor: a cdf is a probability, not a density.
    """
    X = np.asarray(X, dtype=np.float64)
    scalar_input = X.ndim == 0
    X = np.atleast_1d(X)

    Z, shift = normalize_inputs(X, alpha, beta, gamma, delta)

    cdf = alpha_stable_cdf_core(Z, alpha, beta, clip, enforce_monotone)

    return float(cdf[0]) if scalar_input else cdf


def describe_intervals(alpha, beta):
    """
    Readable map of which method covers which interval on x >= 0.
    - Method codes are decoded here, for diagnostics only.
    """
    names = {0: "zolotarev", 1: "numerical", 2: "skorohod"}

    return "  ".join(
        f"({max(lower_bound, 0.0):+.4g},{upper_bound:+.4g}]->{names[method]}"
        for lower_bound, upper_bound, method in generate_switching_intervals(alpha, beta)
    )
