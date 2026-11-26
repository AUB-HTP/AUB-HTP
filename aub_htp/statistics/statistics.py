import math
from functools import lru_cache
from scipy.optimize import minimize, brentq
import aub_htp as ht
from aub_htp.pdf.multivariate import _isotropic_radial_log_pdf, _isotropic_series_radius
from scipy import special
from scipy.integrate import simpson
from scipy.interpolate import PchipInterpolator
from scipy.stats import levy_stable
import numpy as np


def _settle_alpha(alpha: float) -> float:
    """
    Check alpha and return it as a float, set to exactly 1 or 2 when close to them.

    Every function here calls this, so they all take the same branch for the same alpha.
    """
    alpha = float(alpha)
    if np.isclose(alpha, 1):
        return 1.0
    if np.isclose(alpha, 2):
        return 2.0
    if not (0 < alpha <= 2):
        raise ValueError(f"alpha must be in (0, 2], got {alpha!r}.")
    return alpha


def _validate_data(data: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return the data as (n_samples, n_features), and whether it was given as 1-D."""
    data = np.asarray(data, dtype=float)
    if data.ndim not in (1, 2):
        raise ValueError(
            "data must have shape (n_samples,) or (n_samples, n_features), "
            f"got {data.ndim}D data of shape {data.shape}."
        )
    if data.size == 0:
        raise ValueError(f"data must not be empty, got shape {data.shape}.")
    if not np.all(np.isfinite(data)):
        raise ValueError("data must not have NaN or infinite values.")
    data_is_one_dimensional = data.ndim == 1
    if data_is_one_dimensional:
        data = data.reshape(-1, 1)
    return data, data_is_one_dimensional


def _log_unit_sphere_area(dimensions: int) -> float:
    """
    ln |S^(d-1)|, the log of the area of the unit sphere in d dimensions.

    |S^0| = 2, |S^1| = 2 pi, and |S^(d-1)| = 2 pi / (d - 2) * |S^(d-3)|, so

        |S^(d-1)| = 2^(n + 1) pi^floor(d / 2) / ((d - 2) (d - 4) ...)

    where n is the number of factors of the denominator, which stops at 1 or 2.
    """
    denominator_factors = np.arange(dimensions - 2, 0, -2)
    return float(
        (denominator_factors.size + 1) * np.log(2)
        + (dimensions // 2) * np.log(np.pi)
        - np.sum(np.log(denominator_factors))
    )


class _RadialLogPdf:
    """
    ln f_d(r) of the isotropic alpha-stable pdf in d >= 2 dimensions, for alpha != 1, 2.

    _isotropic_radial_log_pdf is too slow to call on the data at every step of a search, so
    it is tabulated once on [r_low, r_join] and interpolated. Beyond r_join it uses its
    series in 1 / r, which is fast, so it is called directly. Below r_low the pdf is f_d(0).
    """

    def __init__(self, dimensions: int, alpha: float):
        self.dimensions = dimensions
        self.alpha = alpha
        # exp(-(gamma |t|)^alpha) = exp(-|t|^alpha / alpha)
        self.gamma_scale = gamma_scale = (1 / alpha) ** (1 / alpha)

        self.radius_join = _isotropic_series_radius(alpha, dimensions, gamma_scale)
        if not np.isfinite(self.radius_join):
            raise ValueError(
                f"The isotropic pdf cannot be computed accurately for alpha = {alpha} in {dimensions} dimensions."
            )
        # f_d(r) = f_d(0) (1 - (r / core)^2 + ...) near 0, so f_d(r_low) = f_d(0) to 1e-6.
        core = np.sqrt(2 * dimensions) * gamma_scale * np.exp(
            0.5 * (math.lgamma(dimensions / alpha) - math.lgamma((dimensions + 2) / alpha))
        )
        self.radius_low = 1e-3 * min(core, self.radius_join)

        n_points = int(60 * np.log10(self.radius_join / self.radius_low)) + 2
        radius = np.geomspace(self.radius_low, self.radius_join, n_points)
        # The pdf decreases with r. Rounding must not break this, since alpha_power relies on it.
        self.log_pdf = np.minimum.accumulate(self._log_pdf(radius))
        self.interpolator = PchipInterpolator(np.log(radius), self.log_pdf)
        self.entropy = self._entropy()

    def _log_pdf(self, radius: np.ndarray) -> np.ndarray:
        return _isotropic_radial_log_pdf(radius, self.alpha, self.dimensions, self.gamma_scale)

    def __call__(self, radius: np.ndarray) -> np.ndarray:
        radius = np.asarray(radius, dtype=float)
        tail = radius > self.radius_join
        out = np.empty(radius.shape, dtype=float)
        out[~tail] = self.interpolator(np.log(np.maximum(radius[~tail], self.radius_low)))
        if np.any(tail):
            out[tail] = np.minimum(self._log_pdf(radius[tail]), self.log_pdf[-1])
        return out

    def _entropy(self) -> float:
        """
        h = -int_0^inf |S^(d-1)| r^(d-1) f_d(r) ln f_d(r) dr

        integrated over s = ln r, where r^(d-1) dr = r^d ds.
        """
        d = self.dimensions
        log_area = _log_unit_sphere_area(d)
        log_low, log_join = np.log(self.radius_low), np.log(self.radius_join)
        # The integrand decays like s exp(-alpha s): stop where exp(-alpha s) has lost a factor
        # exp(-50), but not beyond ln r = 700, where r overflows.
        log_high = max(min(log_join + 50 / self.alpha, 700.0), log_join + 1)
        s = np.linspace(log_low, log_high, 2 * int((log_high - log_low) / 0.02) + 3)
        log_pdf = self(np.exp(s))
        # On [0, r_low] the pdf is f_d(0), and the ball has volume |S^(d-1)| r^d / d.
        inner_mass = np.exp(log_area + d * log_low + self.log_pdf[0]) / d
        return float(
            -inner_mass * self.log_pdf[0]
            - simpson(np.exp(log_area + d * s + log_pdf) * log_pdf, x=s)
        )


@lru_cache(maxsize=None)
def _radial_log_pdf(dimensions: int, alpha: float) -> _RadialLogPdf:
    return _RadialLogPdf(dimensions, alpha)


def isotropic_pdf(data: np.ndarray, alpha: float) -> np.ndarray:
    """
    Isotropic PDF for alpha-stable distribution, one value per sample.

    data has shape (n_samples,) for one feature, or (n_samples, n_features).

    Note for maintainers: This function uses the excplicit formulas for the Cauchy and Normal distributions for alpha = 1 and 2.
    This is for better performance.
    """
    alpha = _settle_alpha(alpha)
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.ndim != 2:
        raise ValueError(
            "data must have shape (n_samples,) or (n_samples, n_features), "
            f"got {data.ndim}D data of shape {data.shape}."
        )
    _, dimensions = data.shape
    return np.exp(_isotropic_log_pdf(np.linalg.norm(data, axis=1), dimensions, alpha))


def _isotropic_log_pdf(radius: np.ndarray, dimensions: int, alpha: float) -> np.ndarray:
    """ln of the isotropic pdf at points of norm `radius`. alpha comes from _settle_alpha."""
    if alpha == 2:
        # Normal distribution: N(0, I)
        return -dimensions / 2 * np.log(2 * np.pi) - radius ** 2 / 2
    elif alpha == 1:
        # Cauchy distribution
        log_coefficient = math.lgamma((dimensions + 1) / 2) - (dimensions + 1) / 2 * np.log(np.pi)
        return log_coefficient - (dimensions + 1) / 2 * np.log1p(radius ** 2)
    elif dimensions == 1:
        scale = (1 / alpha) ** (1 / alpha)
        # The pdf underflows to 0 far in the tail, which gives -inf.
        with np.errstate(divide="ignore"):
            return np.log(ht.alpha_stable.pdf(radius, alpha, 0, loc = 0, scale = scale))
    else:
        return _radial_log_pdf(dimensions, alpha)(radius)


def isotropic_entropy(dimensions: int, alpha: float) -> float:
    """Differential entropy of the isotropic alpha-stable distribution."""
    return _isotropic_entropy(int(dimensions), _settle_alpha(alpha))


@lru_cache(maxsize=None)
def _isotropic_entropy(dimensions: int, alpha: float) -> float:
    # alpha comes from _settle_alpha, so values close to 1 or 2 share one cache entry.
    if alpha == 2:
        # Normal distribution: N(0, I)
        return float(0.5 * dimensions * np.log(2 * np.pi * np.e))
    elif alpha == 1:
        # Cauchy distribution
        return float(
            np.log(np.pi ** ((dimensions + 1) / 2) / special.gamma((dimensions + 1) / 2))
            + ((dimensions + 1) / 2)
            * (special.digamma((dimensions + 1) / 2) - special.digamma(0.5))
        )
    elif dimensions == 1:
        return float(ht.alpha_stable.entropy(alpha, 0, loc = 0, scale = (1 / alpha) ** (1 / alpha)))
    else:
        return _radial_log_pdf(dimensions, alpha).entropy


@lru_cache(maxsize=None)
def _entropy_radius(dimensions: int, alpha: float) -> float:
    """
    The radius x_h > 0 where -ln p = h, for p the isotropic pdf and h its entropy.

    -ln p increases with the radius from -ln p(0) < h, so x_h is found by binary search
    over [0, r_hi], with r_hi = (e^h / V_d)^(1/d) and V_d the volume of the unit ball:
    p(x) V_d |x|^d <= 1 since p is at least p(x) on the ball of radius |x|, so
    -ln p >= ln(V_d r_hi^d) = h at r_hi.

    alpha comes from _settle_alpha, so x_h matches the branch the pdf takes.
    """
    entropy = isotropic_entropy(dimensions, alpha)
    # V_d = |S^(d-1)| / d
    log_ball_volume = _log_unit_sphere_area(dimensions) - np.log(dimensions)
    low, high = 0.0, float(np.exp((entropy - log_ball_volume) / dimensions))
    # 100 halvings bring high - low down to the rounding error of x_h.
    for _ in range(100):
        middle = (low + high) / 2
        if -_isotropic_log_pdf(np.array([middle]), dimensions, alpha)[0] < entropy:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def alpha_power(data: np.ndarray, alpha: float, *, tol: float = 2.0 ** -20, max_iter: int = 40) -> float:
    """
    The alpha-power P of the data: the root of

        g(P) = -mean_k ln p(x_k / P) - h

    for p the isotropic pdf and h its entropy. g decreases with P, so the root is found
    by binary search. With x_h the radius where -ln p = h:

        a = min_k |x_k| / x_h  gives |x_k| / a >= x_h for every k, so g(a) >= 0
        b = max_k |x_k| / x_h  gives |x_k| / b <= x_h for every k, so g(b) <= 0

    The search over [a, b] stops when b - a <= tol, or after max_iter iterations. tol is
    absolute, so it should fit the scale of the data.

    data has shape (n_samples,) for one feature, or (n_samples, n_features).
    """
    alpha = _settle_alpha(alpha)
    data, _ = _validate_data(data)
    _, dimensions = data.shape
    radius = np.linalg.norm(data, axis=1)

    # For all-zero data, g(P) = -ln p(0) - h < 0 for every P.
    # There is no root, so the power is 0.
    if not np.any(radius):
        return 0.0

    if alpha == 2:
        return float(np.sqrt(np.mean(radius ** 2) / dimensions))

    entropy = isotropic_entropy(dimensions, alpha)
    entropy_radius = _entropy_radius(dimensions, alpha)
    low, high = float(radius.min() / entropy_radius), float(radius.max() / entropy_radius)
    for _ in range(max_iter):
        if high - low <= tol:
            break
        # Never at low, which is 0 if some x_k = 0.
        middle = (low + high) / 2
        if -np.mean(_isotropic_log_pdf(radius / middle, dimensions, alpha)) > entropy:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def deprecated_alpha_location(data: np.ndarray, alpha: float) -> np.ndarray:
    data = np.asarray(data)
    data_is_one_dimensional = data.ndim == 1
    if data_is_one_dimensional:
        data = data.reshape(-1, 1)
    optimal_location = minimize(
        lambda location: alpha_power(data - location, alpha),
        x0 = np.median(data, axis = 0),
        method = "Powell",
        options = {
            "maxiter": int(5000),
            "xtol": float(1e-8),
            "ftol": float(1e-8),
        },
    ).x
    if data_is_one_dimensional:
        return optimal_location.item()
    else:
        return optimal_location


def alpha_location(data: np.ndarray, alpha: float) -> np.ndarray:
    alpha = _settle_alpha(alpha)
    data, data_is_one_dimensional = _validate_data(data)

    _, dimensions = data.shape

    entropy = isotropic_entropy(dimensions, alpha)
    epsilon = np.finfo(float).eps
    penalty = 1e6

    def objective_function(log_power_and_location: np.ndarray) -> float: #TODO: ask profs about this objective function
        power = log_power_and_location[0]
        location = log_power_and_location[1:]
        if power < 0:
            return np.inf

        pdf = np.maximum(isotropic_pdf((data - location) / power, alpha), epsilon)
        log_pdf = np.log(pdf)
        residual = np.mean(log_pdf) + entropy

        return power + penalty * residual**2

    optimal_log_power_and_location = minimize(
        objective_function,
        x0 = np.concatenate(([1.0], np.median(data, axis = 0))),
        method = "Powell",
        options = {
            "maxiter": int(5000),
            "xtol": float(1e-8),
            "ftol": float(1e-8),
        },
    ).x

    optimal_location = optimal_log_power_and_location[1:]
    if data_is_one_dimensional:
        return optimal_location.item()
    else:
        return optimal_location
