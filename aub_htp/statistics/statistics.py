import warnings
from functools import lru_cache
from scipy.optimize import minimize, brentq
from aub_htp.pdf.multivariate import _isotropic_radial_log_pdf, _isotropic_series_radius
from scipy import special
from scipy.integrate import simpson
from scipy.interpolate import PchipInterpolator
from scipy.stats import levy_stable
import numpy as np


def _settle_alpha(alpha: float) -> float:
    """Check alpha and return it as a float, set to exactly 1 or 2 when close to them."""
    alpha = float(alpha)
    # Above 2, only a rounding error is accepted.
    if not (0 < alpha <= 2 + 1e-8):
        raise ValueError(f"alpha must be in (0, 2], got {alpha!r}.")
    if np.isclose(alpha, 1):
        return 1.0
    if np.isclose(alpha, 2):
        return 2.0
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

        |S^(d-1)| = 2^(n + 1) pi^floor(d / 2) / ((d - 2) (d - 4) ...)
    """
    denominator_factors = np.arange(dimensions - 2, 0, -2)
    return float(
        (denominator_factors.size + 1) * np.log(2)
        + (dimensions // 2) * np.log(np.pi)
        - np.sum(np.log(denominator_factors))
    )


class _RadialLogPdf:
    """
    Table of ln f_d(r), the isotropic alpha-stable log pdf in d dimensions, alpha != 1, 2.

    Interpolated over ln r on [r_low, r_high]. f_d(0) below, the power law r^-(d + alpha) beyond.
    """

    def __init__(self, dimensions: int, alpha: float):
        self.dimensions = dimensions
        self.alpha = alpha
        # exp(-(gamma |t|)^alpha) = exp(-|t|^alpha / alpha), with gamma = (1 / alpha)^(1 / alpha).
        # The table is built for gamma = 1, over u = r / gamma: f_d(r) = gamma^-d f_d,1(r / gamma).
        self.log_gamma_scale = -np.log(alpha) / alpha

        radius_join = _isotropic_series_radius(alpha, dimensions, 1.0)
        if not np.isfinite(radius_join) or radius_join == 0:
            raise ValueError(
                f"The isotropic pdf cannot be computed accurately for alpha = {alpha} in {dimensions} dimensions."
            )
        # f_d(u) = f_d(0) (1 - (u / core)^2 + ...) near 0
        log_core = 0.5 * (
            np.log(2 * dimensions) + special.gammaln(dimensions / alpha) - special.gammaln((dimensions + 2) / alpha)
        )
        self.log_radius_low = np.log(1e-3) + min(log_core, np.log(radius_join))
        # The mass beyond u decays like u^-alpha: stop at exp(-50) of it, and at ln(u / 2) = 100.
        log_radius_limit = np.log(2) + 100.0
        self.log_radius_high = max(min(np.log(radius_join) + 50 / alpha, log_radius_limit), np.log(radius_join) + 1)

        # Fail before the long build: a very small alpha underflows at the low end.
        ends = np.exp([self.log_radius_low, self.log_radius_high])
        if not np.all(np.isfinite(_isotropic_radial_log_pdf(ends, alpha, dimensions, 1.0))):
            raise ValueError(
                f"The isotropic pdf cannot be computed accurately for alpha = {alpha} in {dimensions} dimensions."
            )

        # About 60 points per decade.
        n_points = int(26 * (self.log_radius_high - self.log_radius_low)) + 2
        self.log_radius = np.linspace(self.log_radius_low, self.log_radius_high, n_points)
        # alpha_power needs the pdf to decrease with u.
        self.log_pdf = np.minimum.accumulate(
            _isotropic_radial_log_pdf(np.exp(self.log_radius), alpha, dimensions, 1.0)
        )
        self.interpolator = PchipInterpolator(self.log_radius, self.log_pdf)
        self.entropy = self._entropy()

    def __call__(self, radius: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            log_radius = np.maximum(np.log(radius) - self.log_gamma_scale, self.log_radius_low)
        beyond = np.maximum(log_radius - self.log_radius_high, 0.0)
        return (
            self.interpolator(log_radius - beyond)
            - (self.dimensions + self.alpha) * beyond
            - self.dimensions * self.log_gamma_scale
        )

    def _entropy(self) -> float:
        """
        The entropy h = -int |S^(d-1)| r^(d-1) f_d(r) ln f_d(r) dr, integrated over the table's points.
        """
        d = self.dimensions
        log_area = _log_unit_sphere_area(d)
        mass_density = np.exp(log_area + d * self.log_radius + self.log_pdf)
        # Mass of the ball of radius r_low, where the pdf is f_d(0).
        inner_mass = mass_density[0] / d
        # Beyond r_high, ln f_d = ln f_d(r_high) - (d + alpha) (s - s_high), and the mass
        # density is its value at r_high times exp(-alpha (s - s_high)).
        alpha = self.alpha
        outer = mass_density[-1] * (self.log_pdf[-1] / alpha - (d + alpha) / alpha ** 2)
        # The table is for gamma = 1: the scale adds d ln gamma.
        return float(
            -inner_mass * self.log_pdf[0]
            - simpson(mass_density * self.log_pdf, x=self.log_radius)
            - outer
            + d * self.log_gamma_scale
        )


@lru_cache(maxsize=32)
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
    data, _ = _validate_data(data)
    _, dimensions = data.shape
    return np.exp(_isotropic_log_pdf(np.linalg.norm(data, axis=1), dimensions, alpha))


def _isotropic_log_pdf(radius: np.ndarray, dimensions: int, alpha: float) -> np.ndarray:
    """ln of the isotropic pdf at points of norm `radius`."""
    if alpha == 2:
        # Normal distribution: N(0, I)
        return -dimensions / 2 * np.log(2 * np.pi) - radius ** 2 / 2
    elif alpha == 1:
        # Cauchy distribution
        log_coefficient = special.gammaln((dimensions + 1) / 2) - (dimensions + 1) / 2 * np.log(np.pi)
        return log_coefficient - (dimensions + 1) / 2 * np.log1p(radius ** 2)
    else:
        return _radial_log_pdf(dimensions, alpha)(radius)


def isotropic_entropy(dimensions: int, alpha: float) -> float:
    """Differential entropy of the isotropic alpha-stable distribution."""
    return _isotropic_entropy(int(dimensions), _settle_alpha(alpha))


@lru_cache(maxsize=None)
def _isotropic_entropy(dimensions: int, alpha: float) -> float:
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
    else:
        return _radial_log_pdf(dimensions, alpha).entropy


@lru_cache(maxsize=None)
def _entropy_radius(dimensions: int, alpha: float) -> float:
    """
    The radius x_h where -ln p = h, for p the isotropic pdf and h its entropy.

    Binary search over [0, r_hi], with r_hi = (e^h / V_d)^(1/d) and V_d the volume of the unit ball.
    """
    entropy = isotropic_entropy(dimensions, alpha)
    # V_d = |S^(d-1)| / d
    log_ball_volume = _log_unit_sphere_area(dimensions) - np.log(dimensions)
    low, high = 0.0, float(np.exp((entropy - log_ball_volume) / dimensions))
    # Enough halvings to reach the rounding error of x_h.
    for _ in range(100):
        middle = (low + high) / 2
        if -_isotropic_log_pdf(np.array([middle]), dimensions, alpha)[0] < entropy:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def alpha_power(data: np.ndarray, alpha: float, *, tol: float = 2.0 ** -20, max_iter: int = 100) -> float:
    """
    The alpha-power P of the data: the root of

        g(P) = -mean_k ln p(x_k / P) - h

    for p the isotropic pdf and h its entropy.

    Binary search over [min_k |x_k| / x_h, max_k |x_k| / x_h], with x_h the radius where
    -ln p = h. It stops when the interval is at most tol times its upper end, or after
    max_iter iterations. tol is relative, so it does not depend on the scale of the data.

    A sample far from the others takes more iterations: about log2(max_k |x_k| / (x_h P))
    more than -log2(tol). A RuntimeWarning is given if max_iter ends the search before tol.

    data has shape (n_samples,) for one feature, or (n_samples, n_features).
    """
    alpha = _settle_alpha(alpha)
    data, _ = _validate_data(data)
    _, dimensions = data.shape
    radius = np.linalg.norm(data, axis=1)

    # g(P) < 0 for every P: no root.
    if not np.any(radius):
        return 0.0

    if alpha == 2:
        return float(np.sqrt(np.mean(radius ** 2) / dimensions))

    entropy = isotropic_entropy(dimensions, alpha)
    entropy_radius = _entropy_radius(dimensions, alpha)
    low, high = float(radius.min() / entropy_radius), float(radius.max() / entropy_radius)
    for _ in range(max_iter):
        if high - low <= tol * high:
            break
        middle = (low + high) / 2
        if not (low < middle < high):
            # No float is left between the ends.
            break
        if -np.mean(_isotropic_log_pdf(radius / middle, dimensions, alpha)) > entropy:
            low = middle
        else:
            high = middle
    else:
        if high - low > tol * high:
            warnings.warn(
                f"alpha_power stopped after max_iter = {max_iter} iterations with a relative "
                f"error of {(high - low) / high:.1e}, above tol = {tol:.1e}. Increase max_iter.",
                RuntimeWarning,
                stacklevel=2,
            )
    return (low + high) / 2


def alpha_location(
    data: np.ndarray,
    alpha: float,
    *,
    tol: float = 2.0 ** -20,
    power_tol: float | None = None,
    max_iter: int = 100,
) -> np.ndarray:
    """
    The alpha-location of the data: arg min over mu of alpha_power(data - mu).

    Powell search in the box of the data, starting from the median. A constant feature
    gets its value. For alpha = 2 the location is the mean.

    tol is the tolerance for mu, relative to the power of the data around its median.
    power_tol and max_iter are passed to alpha_power. The tolerances are relative, so
    they do not depend on the scale of the data.
    mu cannot be more accurate than about sqrt(power_tol) times the power, so power_tol
    is tol^2 by default.

    data has shape (n_samples,) for one feature, or (n_samples, n_features).
    """
    alpha = _settle_alpha(alpha)
    data, data_is_one_dimensional = _validate_data(data)
    relative_tol = float(np.clip(tol, np.finfo(float).eps, 1.0))
    if power_tol is None:
        power_tol = relative_tol ** 2

    if alpha == 2:
        location = np.mean(data, axis=0)
    else:
        low, high = np.min(data, axis=0), np.max(data, axis=0)
        location = low.copy()
        searched = high > low
        if np.any(searched):
            start = np.median(data[:, searched], axis=0)

            def power(offset: np.ndarray) -> float:
                location[searched] = start + unit * offset
                return alpha_power(data - location, alpha, tol=power_tol, max_iter=max_iter)

            # mu = start + unit * offset, with unit the power at the start, so that
            # Powell's tolerances are relative to the scale of the data.
            unit = 1.0
            unit = power(np.zeros_like(start))
            offset = minimize(
                power,
                x0 = np.zeros_like(start),
                method = "Powell",
                bounds = list(zip((low[searched] - start) / unit, (high[searched] - start) / unit)),
                options = {
                    "xtol": relative_tol,
                    "ftol": relative_tol ** 2,
                },
            ).x
            location[searched] = start + unit * offset

    if data_is_one_dimensional:
        return location.item()
    else:
        return location
