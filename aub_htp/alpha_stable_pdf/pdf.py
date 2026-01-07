from .skorohod import skorohod_formula_1, skorohod_formula_3
from .zolotarev import generate_pdf_zolotarev_1, generate_pdf_NolanS0
import numpy as np
import pickle
from scipy.integrate import quad_vec


def remove_left_monotonicity_spikes(x_vals, pdf_vals):
    """
    Fix non-monotone bumps on the rising side (left of the mode).
    Strategy:
    - Find the mode index.
    - Scan leftwards. Whenever f[i] >= f[i+1] (violation of strict increase),
      locate a left boundary where the increase resumes and a right boundary
      where increase resumes after the bump.
    - Linearly interpolate between those two anchors.
    """
    pdf_vals = pdf_vals.copy()
    peak_idx = np.argmax(pdf_vals)
    i = peak_idx - 1

    while i > 0:
        # Violation: not strictly increasing toward the mode
        if pdf_vals[i] >= pdf_vals[i + 1]:
            # walk left to find last increasing point
            good_left = i
            while good_left > 1 and pdf_vals[good_left - 1] >= pdf_vals[good_left]:
                good_left -= 1
            good_left -= 1  # step one more to land on a clean anchor

            # walk right (but still left of the peak) to find first increase
            good_right = i + 1
            while good_right < peak_idx - 1 and pdf_vals[good_right] >= pdf_vals[good_right + 1]:
                good_right += 1
            good_right += 1  # step to include first increasing index

            # Only interpolate if we have a span
            if good_left >= 0 and good_right < peak_idx and good_right > good_left + 1:
                x1, y1 = x_vals[good_left], pdf_vals[good_left]
                x2, y2 = x_vals[good_right], pdf_vals[good_right]
                for j in range(good_left + 1, good_right):
                    pdf_vals[j] = np.interp(x_vals[j], [x1, x2], [y1, y2])

            # continue scanning from the new left anchor
            i = good_left
        else:
            i -= 1

    return pdf_vals


def remove_all_monotonicity_spikes(x_vals, pdf_vals):
    """
    Fix bumps on both sides of the mode.
    - Clean the left side by direct scan.
    - Reverse, reuse the same routine, then flip back to fix the right side.
    """
    pdf_vals = remove_left_monotonicity_spikes(x_vals, pdf_vals)

    pdf_vals_reversed = pdf_vals[::-1]
    pdf_vals = remove_left_monotonicity_spikes(x_vals, pdf_vals_reversed)

    pdf_vals = np.flip(pdf_vals)
    return pdf_vals


def normalize_inputs(X, alpha, beta, gamma, delta):
    """
    Normalize to unit scale for S1-style behavior.
    - When alpha==1, apply the log shift to preserve centering.
    - Return Z = (X - shift)/gamma and the shift used (for diagnostics).
    """
    if alpha == 1:
        shift = delta + (2 / np.pi) * beta * gamma * np.log(gamma)
    else:
        shift = delta
    Z = (X - shift) / gamma
    return Z, shift


from importlib.resources import files as _res_files
from pathlib import Path
import pickle


def _load_pickle(name: str):
    """
    Load an installed pickle resource.
    Expects package data under 'generate_stable_pdf.data'.
    """
    with open(Path(__file__).parent / "data" / name, 'rb') as f:
        return pickle.load(f)


def generate_pdf_alpha_less_1(X, alpha, beta):
    """
    Piecewise pdf for 0 < alpha < 1.
    - Choose among:
      • Skorohod tail series (formula 1) for large |x|
      • Zolotarev integral (Type B) for mid range
      • CF integral in S0 for near zero
    - Thresholds (x_min) are read from pickled interpolators per (alpha,beta).
    - Handle x<0 by reflection with beta -> -beta.
    - Force pdf(0)=0 when beta is ±1 (one-sided support in the limit).
    """
    lowest_x_skorohod_fn = _load_pickle("skorohod_1_interpolator_x_min.pkl")
    lowest_x_zolotarev_fn = _load_pickle("zolotarev_interpolator_x_min_alpha_less_1.pkl")
    pdf = np.zeros_like(X)

    # Right side (x >= 0)
    if beta != -1:
        lowest_x_zolotarev = lowest_x_zolotarev_fn([(alpha, beta)])[0]
        lowest_x_skorohod = max(lowest_x_skorohod_fn([(alpha, beta)])[0], lowest_x_zolotarev)

        mask_pos_sk = X >= lowest_x_skorohod
        mask_pos_zo = (X >= lowest_x_zolotarev) & (X < lowest_x_skorohod)
        mask_pos_S0 = (X < lowest_x_zolotarev) & (X >= 0)

        # Near α≈1 the Zolotarev mid-range becomes less reliable; prefer S0
        if alpha > 0.95:
            mask_pos_zo = np.zeros_like(X, dtype=bool)
            mask_pos_S0 = (X >= 0) & (X < lowest_x_skorohod)

        if np.any(mask_pos_sk):
            pdf[mask_pos_sk] = skorohod_formula_1(X[mask_pos_sk], alpha, beta)
        if np.any(mask_pos_zo):
            pdf[mask_pos_zo] = generate_pdf_zolotarev_1(X[mask_pos_zo], alpha, beta)
        if np.any(mask_pos_S0):
            pdf[mask_pos_S0] = generate_pdf_NolanS0(X[mask_pos_S0], alpha, beta)

    # Left side (x < 0) via reflection
    mask_neg = X < 0
    if np.any(mask_neg):
        if beta != 1:
            x_reflected = np.abs(X[mask_neg])
            beta_flipped = -beta

            lowest_x_skorohod = lowest_x_skorohod_fn([(alpha, beta_flipped)])[0]
            lowest_x_zolotarev = lowest_x_zolotarev_fn([(alpha, beta_flipped)])[0]

            mask_neg_sk = x_reflected >= lowest_x_skorohod
            mask_neg_zo = (x_reflected >= lowest_x_zolotarev) & (x_reflected < lowest_x_skorohod)
            mask_neg_S0 = (x_reflected < lowest_x_zolotarev) & (x_reflected >= 0)

            if alpha > 0.95:
                mask_neg_zo = np.zeros_like(x_reflected, dtype=bool)
                mask_neg_S0 = (x_reflected >= 0) & (x_reflected < lowest_x_skorohod)

            pdf_reflected = np.zeros_like(x_reflected)

            if np.any(mask_neg_sk):
                pdf_reflected[mask_neg_sk] = skorohod_formula_1(x_reflected[mask_neg_sk], alpha, beta_flipped)
            if np.any(mask_neg_zo):
                pdf_reflected[mask_neg_zo] = generate_pdf_zolotarev_1(x_reflected[mask_neg_zo], alpha, beta_flipped)
            if np.any(mask_neg_S0):
                pdf_reflected[mask_neg_S0] = generate_pdf_NolanS0(x_reflected[mask_neg_S0], alpha, beta_flipped)

            pdf[mask_neg] = pdf_reflected

    # One-sided edge case at zero for β=±1
    if beta == 1 or beta == -1:
        mask_zero = X == 0
        pdf[mask_zero] = 0

    return pdf


def generate_pdf_alpha_equal_1(X, beta):
    """
    α = 1 pdf via characteristic function integral (vectorized).
    f(x) = (1/π) ∫_0^∞ e^{-t} cos( x t + (2/π) β t log t ) dt
    """
    X = np.asarray(X, dtype=np.float64)

    def integrand(t, x):
        return np.exp(-t) * np.cos(x * t + (2 / np.pi) * beta * t * np.log(t))

    val, _ = quad_vec(integrand, 0, np.inf, args=(X,), epsabs=1e-12, epsrel=1e-12, limit=100)
    return val / np.pi


def generate_pdf_alpha_greater_1(X, alpha, beta):
    """
    Piecewise pdf for 1 < alpha ≤ 2.
    - Choose among:
      • Skorohod tail series (formula 3) with N terms
      • S0 CF integral for mid and near-zero ranges
      • Optional Zolotarev region; near α≈1 prefer S0 for stability
    - For x<0 reflect with beta -> -beta and reuse logic.
    """
    lowest_x_skorohod_fn = _load_pickle("skorohod_3_interpolator_x_min.pkl")
    lowest_x_zolotarev_fn = _load_pickle("zolotarev_interpolator_x_min_alpha_greater_1.pkl")

    pdf = np.zeros_like(X)

    lowest_x_zolotarev = lowest_x_zolotarev_fn([(alpha, beta)])[0]
    lowest_x_skorohod = max(lowest_x_skorohod_fn([(alpha, beta)])[0], lowest_x_zolotarev)

    mask_pos_sk = X >= lowest_x_skorohod
    mask_pos_zo = (X >= lowest_x_zolotarev) & (X < lowest_x_skorohod)
    mask_pos_S0 = (X < lowest_x_zolotarev) & (X >= 0)

    # Close to α=1 switch Zolotarev mid-range off, keep S0
    if alpha < 1.05:
        mask_pos_zo = np.zeros_like(X, dtype=bool)
        mask_pos_S0 = (X >= 0) & (X < lowest_x_skorohod)

    # Tail series terms; larger α needs fewer terms
    N = int(120 // alpha)

    if np.any(mask_pos_sk):
        pdf[mask_pos_sk] = skorohod_formula_3(X[mask_pos_sk], alpha, beta, N)
    if np.any(mask_pos_zo):
        pdf[mask_pos_zo] = generate_pdf_NolanS0(X[mask_pos_zo], alpha, beta)
    if np.any(mask_pos_S0):
        pdf[mask_pos_S0] = generate_pdf_NolanS0(X[mask_pos_S0], alpha, beta)

    # Left side by reflection
    mask_neg = X < 0
    if np.any(mask_neg):
        x_reflected = np.abs(X[mask_neg])
        beta_flipped = -beta

        lowest_x_skorohod = lowest_x_skorohod_fn([(alpha, beta_flipped)])[0]
        lowest_x_zolotarev = lowest_x_zolotarev_fn([(alpha, beta_flipped)])[0]

        mask_neg_sk = x_reflected >= lowest_x_skorohod
        mask_neg_zo = (x_reflected >= lowest_x_zolotarev) & (x_reflected < lowest_x_skorohod)
        mask_neg_S0 = (x_reflected < lowest_x_zolotarev) & (x_reflected >= 0)

        if alpha < 1.05:
            mask_neg_zo = np.zeros_like(x_reflected, dtype=bool)
            mask_neg_S0 = (x_reflected >= 0) & (x_reflected < lowest_x_skorohod)

        pdf_reflected = np.zeros_like(x_reflected)

        if np.any(mask_neg_sk):
            pdf_reflected[mask_neg_sk] = skorohod_formula_3(x_reflected[mask_neg_sk], alpha, beta_flipped, N)
        if np.any(mask_neg_zo):
            pdf_reflected[mask_neg_zo] = generate_pdf_NolanS0(x_reflected[mask_neg_zo], alpha, beta_flipped)
        if np.any(mask_neg_S0):
            pdf_reflected[mask_neg_S0] = generate_pdf_NolanS0(x_reflected[mask_neg_S0], alpha, beta_flipped)

        pdf[mask_neg] = pdf_reflected

    return pdf


def pad_grid(X, left_pts=5, right_pts=5, growth=1.08):
    """
    Geometric padding for spike smoothing at boundaries.
    - Extend the grid on both sides using a geometric step growth.
    - Return:
      • X_pad: padded grid (ascending)
      • core_slice: slice to map back to original region
    """
    X = np.asarray(X, dtype=np.float64)
    assert X.ndim == 1 and X.size >= 3, "X must be 1D with >=3 points"
    if not np.all(np.diff(X) > 0):
        raise ValueError("X must be strictly increasing")

    # Right padding
    dx_r = X[-1] - X[-2]
    steps_r = dx_r * np.cumprod(np.full(right_pts, growth))
    right_ext = X[-1] + np.cumsum(steps_r)

    # Left padding
    dx_l = X[1] - X[0]
    steps_l = dx_l * np.cumprod(np.full(left_pts, growth))
    left_ext = X[0] - np.cumsum(steps_l)
    left_ext = left_ext[::-1]  # keep ascending order

    X_pad = np.concatenate([left_ext, X, right_ext])
    core_slice = slice(left_pts, left_pts + len(X))
    return X_pad, core_slice


def alpha_stable_pdf_core(X, alpha, beta, gamma, delta, ):
    """
    Core density on the normalized grid (unit scale).
    Steps:
    - Normalize inputs to S1-like domain.
    - Compute piecewise pdf by alpha regime.
    - Apply spike removal on the full normalized domain.
    - Zero-out the forbidden side for extreme skew in α<1.
    - Return scaled density (divide by gamma).
    """
    X = np.array(X, dtype=np.float64)
    X, shift = normalize_inputs(X, alpha, beta, gamma, delta)
    if 0 < alpha and alpha < 1:
        pdf = generate_pdf_alpha_less_1(X, alpha, beta)
    elif alpha == 1:
        pdf = generate_pdf_alpha_equal_1(X, beta)
    elif 1 < alpha and alpha <= 2:
        pdf = generate_pdf_alpha_greater_1(X, alpha, beta)
    else:
        raise Exception("Invalid alpha value")

    pdf = remove_all_monotonicity_spikes(X, pdf)

    # For α<1 and |β|=1, support collapses to one side as x→0.
    if 0 < alpha and alpha < 1:
        if beta == 1:
            mask_X = X <= 0
            pdf[mask_X] = 0
        elif beta == -1:
            mask_X = X >= 0
            pdf[mask_X] = 0

    return pdf / gamma


def generate_alpha_stable_pdf(X, alpha, beta, gamma, delta, pad_left=10, pad_right=10, growth=1.05):
    """
    Public pdf wrapper with boundary padding and spike cleanup.
    Pipeline:
    1) Pad the query grid (stabilizes denoising near edges).
    2) Evaluate normalized core density.
    3) Remove monotonicity spikes on padded grid.
    4) Slice back to the original grid.
    5) Return density.
    """
    X = np.asarray(X, dtype=np.float64)
    X_pad, sl = pad_grid(X, left_pts=pad_left, right_pts=pad_right, growth=growth)

    dens_pad = alpha_stable_pdf_core(X_pad, alpha, beta, gamma, delta)

    # Clean on padded range for smoother edges
    dens_pad = remove_all_monotonicity_spikes(X_pad, dens_pad)

    # Recover original region
    dens = dens_pad[sl]

    return dens


