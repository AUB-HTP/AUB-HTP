"""Numerical multivariate alpha-stable density.

Implements the one-dimensional-projection method of

    M. Matsui and A. Takemura (2006),
    "Integral representations of one dimensional projections for
     multivariate stable densities", arXiv:math/0608570.

The multivariate density is written (Nolan 1998, Theorem 1.1, the A -> A
projection) as an integral over the unit sphere

    f(x) = \\int_{S^{d-1}} g_{a,d}( v(s), beta(s) ) sigma(s)^{-d} ds,

where ``sigma(s)`` and ``beta(s)`` are the scale and skewness of the
one-dimensional projection of the stable vector onto direction ``s``,
determined by the spectral measure ``Gamma`` through equations (4)-(6) of the
paper, and

    v(s) = <x - nu, s> / sigma(s)                                 (alpha != 1)
    v(s) = (<x - nu, s> - mu(s) - (2/pi) beta sigma log sigma) / sigma  (alpha == 1)

``g_{a,d}`` is the projected one dimensional function of definition (1),

    g_{a,d}(v, b) = (2 pi)^{-d} \\int_0^inf cos(v u - b tan(pi a / 2) u^a)
                                            u^{d-1} exp(-u^a) du     (a != 1)
    g_{1,d}(v, b) = (2 pi)^{-d} \\int_0^inf cos(v u + (2/pi) b u log u)
                                            u^{d-1} exp(-u) du       (a == 1)

For ``alpha != 1`` it is evaluated with the finite-interval representation of
Proposition 2.1 / Theorem 2.1, built on ``h^n(x; a, beta)`` (equation (7)).
That representation is non-oscillatory and accurate for every ``v``.

For ``alpha == 1`` Zolotarev's (B) parameterization used by ``h^n`` is singular
as ``beta -> 0`` (it carries ``exp(-x / beta)``), so this module evaluates
``g_{1,d}`` directly instead:

* ``beta == 0``            -- closed form ``(d-1)! (2 pi)^{-d} (1+v^2)^{-d/2}
                              cos(d arctan v)``;
* ``|v| <= _A1_ASYMPTOTIC_V`` -- composite Gauss-Legendre panels sized to the
  oscillation of ``cos(v u + ...)`` (relative error ~1e-12);
* ``|v| >  _A1_ASYMPTOTIC_V`` -- the large-``|v|`` series obtained by expanding
  ``exp(i c u log u)``, ``c = (2/pi) beta``, and integrating term by term,

      g = (2 pi)^{-d} Re sum_k ((i c)^k / k!) d^k/ds^k[ Gamma(s) p^{-s} ]|_{s=d+k}

  with ``p = 1 - i |v|`` (relative error ~3e-7 at ``|v| = 20``, better beyond).
  ``g(v, b) = g(-v, -b)`` folds negative ``v``.

Spectral-measure samples supply the Monte-Carlo estimators

    sigma(t)^a           ~= (mass / N) * sum_i |<t, V_i>|^a
    beta(t) sigma(t)^a   ~= (mass / N) * sum_i sign(<t, V_i>) |<t, V_i>|^a
    mu(t)                ~= -(2/pi) (mass / N) sum_i <t,V_i> log(|<t,V_i>|/|V_i|)

where ``V_i`` are the samples returned by ``sampler.sample`` and
``mass = sampler.mass()``.  The ``mu`` estimator takes the logarithm of the
*unit*-direction projection so the scale carried by the samples does not leak
into the location.

Performance
-----------
Evaluating ``f`` at ``n`` points costs ``n * m`` kernel evaluations for ``m``
sphere nodes, which is the dominant cost.  :class:`MultivariateStableDensity`
therefore

1. computes the sphere nodes and ``sigma``/``beta``/``mu`` **once** (the
   previous implementation re-sampled the spectral measure on every call), and
2. replaces the per-point kernel quadrature with a spline of ``g`` in ``v``
   built from the exact kernel on an ``arcsinh``-spaced grid (a 2-D spline in
   ``(v, beta)`` when ``beta(s)`` varies across directions).

Pass ``exact=True`` to bypass the spline and evaluate the kernel directly; the
test-suite uses that path as the reference.
"""

import math
from collections import OrderedDict

import numpy as np
import numpy.typing as npt
from scipy.integrate import quad_vec
from scipy.interpolate import CubicSpline, RectBivariateSpline
from scipy.special import digamma, jv, polygamma
from scipy.special import gamma as gamma_fn
from scipy.stats import norm, qmc

from aub_htp.random.spectral_measure_sampler import (
    BaseSpectralMeasureSampler,
    DiscreteSampler,
    EllipticSampler,
    IsotropicSampler,
)


_PDF_MODEL_CACHE_MAXSIZE = 8
_pdf_model_cache = OrderedDict()


def _array_key(array: npt.ArrayLike):
    """Return a hashable, value-based key for an array-like argument."""
    value = np.asarray(array)
    return value.dtype.str, value.shape, value.tobytes()


def _measure_key(measure: BaseSpectralMeasureSampler):
    """Return a stable cache key for built-in spectral measures."""
    if isinstance(measure, IsotropicSampler):
        return (
            "isotropic", measure.dimensions(), measure.alpha, measure.gamma,
            measure.mass(),
        )
    if isinstance(measure, EllipticSampler):
        return (
            "elliptical", measure.dimensions(), measure.alpha,
            _array_key(measure.sigma), measure.mass(),
        )
    if isinstance(measure, DiscreteSampler):
        return (
            "discrete", measure.dimensions(), _array_key(measure.positions),
            _array_key(measure.weights), measure.mass(),
        )
    # Custom samplers have no general value representation. The cached model
    # retains the sampler, so its identity cannot be reused while cached.
    return "object", id(measure)


def _random_state_key(random_state):
    if isinstance(random_state, (np.random.RandomState, np.random.Generator)):
        return "object", id(random_state)
    return random_state


def clear_pdf_model_cache():
    """Discard all cached multivariate density models."""
    _pdf_model_cache.clear()

from .zolotarev import theta0_stable  # noqa: F401  (kept for API discoverability)

# Switch from panel quadrature to the large-|v| series for alpha == 1.
_A1_ASYMPTOTIC_V = 30.0
# exp(-u) is below double precision past this, so the alpha == 1 integrand is
# truncated there.
_A1_UPPER = 45.0
# How far inside |beta| = 1 the h-function skewness is clamped (see
# _h_real_imag_general): the endpoint of the finite-interval kernel is singular
# exactly at beta = -1.
_BETA_ENDPOINT_EPS = 1e-9


def _K(alpha: float) -> float:
    return alpha - 1.0 + np.sign(1.0 - alpha)


# --------------------------------------------------------------------------- #
# alpha != 1: Matsui-Takemura finite-interval representation
# --------------------------------------------------------------------------- #
def _h_real_imag(x, beta, alpha, n, *, quad_limit=200):
    """Real and imaginary parts of ``h^n(x; alpha, beta)`` (Theorem 2.1).

    ``x`` and ``beta`` are broadcastable arrays.  ``x`` must be positive and
    ``beta`` is expressed in Zolotarev's (B) parameterization.
    """
    x = np.asarray(x, dtype=np.float64)
    beta = np.broadcast_to(np.asarray(beta, dtype=np.float64), x.shape).copy()
    if alpha == 1.0:
        raise ValueError("alpha == 1 uses the direct representation, not h^n")
    return _h_real_imag_general(x, beta, alpha, n, quad_limit)


def _h_real_imag_general(x, beta, alpha, n, quad_limit):
    K = _K(alpha)

    # At beta = -1 the lower endpoint of the finite-interval kernel becomes
    # singular: there a + a0 = 0, so sin(alpha (a + a0)) = 0 and
    # base^(alpha/(1-alpha)) diverges with a negative cosine, i.e. the damping
    # factor exp(-expo) blows up instead of vanishing.  For alpha < 1 it is
    # worse still: K = alpha makes theta = -1, so span = 1 + theta collapses to
    # zero width and the integral returns exactly 0.  ``g`` is continuous in
    # beta, so clamping just inside the endpoint removes the singularity at a
    # cost far below the quadrature tolerance.
    beta = np.clip(beta, -1.0 + _BETA_ENDPOINT_EPS, 1.0 - _BETA_ENDPOINT_EPS)

    theta = beta * K / alpha            # lower integration limit is -theta
    a0 = (np.pi / 2.0) * theta          # angle pivot theta0
    half = np.pi / 2.0
    p_exp = 1.0 / (1.0 - alpha)
    e_exp = alpha / (1.0 - alpha)

    def _integrand(t):
        # affine map t in [0, 1] -> phi in [-theta, 1]
        span = 1.0 + theta
        phi = -theta + span * t
        a = half * phi
        with np.errstate(all="ignore"):
            sinB = np.sin(alpha * (a + a0))
            cosa = np.cos(a)
            base = sinB / (x * cosa)
            r = base ** p_exp
            rp = p_exp * r * (alpha * np.cos(alpha * (a + a0)) / sinB + np.tan(a))
            expo = base ** e_exp * np.cos((alpha - 1.0) * a + alpha * a0) / cosa
            damp = np.exp(-expo)
            ang = half * (n + 1) * (phi + 1.0)
            sin_ang = np.sin(ang)
            cos_ang = np.cos(ang)
            rn = r ** n
            Vn = rn * (rp * sin_ang + r * cos_ang)
            Wn = rn * (r * sin_ang - rp * cos_ang)
            out = np.stack([0.5 * damp * Vn, 0.5 * damp * Wn]) * span
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    val, _ = quad_vec(_integrand, 0.0, 1.0, limit=quad_limit)
    # The change of variables in Zolotarev's derivation reverses the orientation
    # of the V_n / W_n integral for alpha > 1 (the |alpha - 1| that the 1D
    # formula carries explicitly); restore it here.
    s = np.sign(1.0 - alpha)
    HR, HI = s * val[0], s * val[1]

    # NOTE: earlier revisions added the Theorem 2.1 (c) "extra term"
    #   -(1/pi) int_0^tau exp(x s - s^a) s^n ds,  tau = (a/x)^{1/(1-a)}
    # to Im h whenever alpha < 1 and beta == 1.  Decomposing Im h against the
    # defining integral shows the clamped finite-interval integral above already
    # equals the required value to quadrature accuracy (~1e-9) for every
    # (alpha, beta, x) tested, so adding that term double-counts -- at beta = 1
    # it doubled g for even d, where g depends on Im h alone.  It is therefore
    # deliberately absent.
    return HR, HI


def _g_general_direct(v, beta, alpha, d, *, gl_nodes=20, per_period=2.0,
                      max_panels=400_000):
    """``g_{alpha,d}`` (``alpha != 1``) straight from definition (1).

    Composite Gauss-Legendre on ``[0, u_max]`` with the panel count matched to
    the ``cos(v u - ...)`` oscillation.  ``exp(-u^alpha)`` decays slowly for
    small ``alpha``, so ``u_max`` -- and hence the cost -- grows like
    ``39^(1/alpha)``.  Used only where the finite-interval kernel breaks down
    (``alpha < 1`` at ``|beta| = 1``), so the extra cost is confined.
    """
    v = np.atleast_1d(np.asarray(v, dtype=np.float64))
    beta = np.broadcast_to(np.asarray(beta, dtype=np.float64), v.shape)
    tau = np.tan(np.pi * alpha / 2.0)

    # exp(-u^alpha) < ~1e-17 past this point.
    u_max = float(min(39.0 ** (1.0 / alpha), 1.0e5))
    v_max = max(float(np.max(np.abs(v))), 1e-3)
    n_panels = int(np.ceil(per_period * v_max * u_max / (2.0 * np.pi)))
    n_panels = max(n_panels, 64)
    if n_panels > max_panels:
        import warnings
        warnings.warn(
            f"multivariate stable kernel: alpha={alpha:g} with |beta|=1 needs "
            f"{n_panels} quadrature panels for |v| up to {v_max:g}; capping at "
            f"{max_panels}. Results in this corner may lose accuracy.",
            RuntimeWarning, stacklevel=2,
        )
        n_panels = max_panels

    xg, wg = np.polynomial.legendre.leggauss(gl_nodes)
    edges = np.linspace(0.0, u_max, n_panels + 1)
    half = 0.5 * (edges[1] - edges[0])
    total = np.zeros_like(v)
    for lo, hi in zip(edges[:-1], edges[1:]):
        u = 0.5 * (lo + hi) + half * xg
        ua = u ** alpha
        amp = wg * u ** (d - 1) * np.exp(-ua)
        phase = v[:, None] * u[None, :] - beta[:, None] * (tau * ua)[None, :]
        total += half * (np.cos(phase) @ amp)
    return total / (2.0 * np.pi) ** d


def _g_general(v, beta, alpha, d, quad_limit):
    """``g_{alpha,d}`` for ``alpha != 1`` via the h-function representation."""
    n = d - 1
    tan_h = np.tan(np.pi * alpha / 2.0)
    c0 = 1.0 / np.sqrt(1.0 + (beta * tan_h) ** 2)          # cos(alpha*theta0)
    K = _K(alpha)
    beta_B = (2.0 / (np.pi * K)) * np.arctan(beta * tan_h)
    x = c0 ** (1.0 / alpha) * v
    sgn = np.sign(x)
    sgn = np.where(sgn == 0.0, 1.0, sgn)
    x_abs = np.abs(x)
    beta_star = beta_B * sgn

    HR, HI = _h_real_imag(x_abs, beta_star, alpha, n, quad_limit=quad_limit)

    coef = c0 ** (d / alpha) / (2.0 ** d * np.pi ** (d - 1))
    g = coef * (HR * np.cos(np.pi * (d - 1) / 2.0) + HI * np.sin(np.pi * (d - 1) / 2.0))
    g = np.array(g, dtype=np.float64, copy=True)

    # For alpha < 1 at |beta_B| = 1 the finite-interval kernel is unusable: the
    # lower endpoint sits at a = -pi/2, where cos(a) = 0 makes ``base`` diverge,
    # so ``r`` and ``expo`` overflow and ``damp`` alternates between underflow
    # and overflow.  For small |x| the whole integrand collapses to zero.  Fall
    # back to the defining integral there (exact, just more expensive).
    #
    # Only even ``d`` needs it: cos(pi (d-1)/2) and sin(pi (d-1)/2) select Re h
    # for odd ``d`` and Im h for even ``d``, and the breakdown is confined to
    # Im h -- for odd ``d`` the h-function stays accurate to ~1e-9 there.
    if alpha < 1.0 and d % 2 == 0:
        degenerate = np.abs(beta_star) > 1.0 - 1e-6
        if np.any(degenerate):
            g[degenerate] = _g_general_direct(v[degenerate], beta[degenerate],
                                              alpha, d)

    # x = 0 is a removable singularity of the finite-interval kernel (r has x in
    # its denominator); evaluate definition (1) directly there.
    zero = x_abs <= 1e-9
    if np.any(zero):
        g[zero] = _g_at_zero(beta[zero], alpha, d, quad_limit)
    return g


def _g_at_zero(beta, alpha, d, quad_limit):
    """``g_{alpha,d}(0, beta)`` from definition (1) with ``v = 0``."""
    beta = np.atleast_1d(np.asarray(beta, dtype=np.float64))

    def _integrand(u):
        with np.errstate(all="ignore"):
            out = (np.cos(beta * np.tan(np.pi * alpha / 2.0) * u ** alpha)
                   * u ** (d - 1) * np.exp(-u ** alpha))
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    val, _ = quad_vec(_integrand, 0.0, np.inf, limit=quad_limit)
    return val / (2.0 * np.pi) ** d


# --------------------------------------------------------------------------- #
# alpha == 1: direct evaluation of definition (1)
# --------------------------------------------------------------------------- #
def _g_alpha1_symmetric(v, d):
    """``g_{1,d}(v, 0) = (d-1)!/(2 pi)^d (1+v^2)^{-d/2} cos(d arctan v)``."""
    return (math.factorial(d - 1) / (2.0 * np.pi) ** d
            * (1.0 + v ** 2) ** (-d / 2.0) * np.cos(d * np.arctan(v)))


def _g_alpha1_panels(v, beta, d, *, gl_nodes=20, per_period=2.0,
                     upper=_A1_UPPER):
    """Composite Gauss-Legendre quadrature of ``g_{1,d}`` on ``[0, upper]``.

    The panel count resolves the ``cos(v u + ...)`` oscillation, so accuracy is
    uniform in ``v`` (unlike a single adaptive call, which loses the phase).
    """
    v = np.atleast_1d(np.asarray(v, dtype=np.float64))
    c = (2.0 / np.pi) * np.broadcast_to(np.asarray(beta, dtype=np.float64),
                                        v.shape)
    v_max = max(float(np.max(np.abs(v))), 1e-3)
    n_panels = int(np.clip(np.ceil(per_period * v_max * upper / (2.0 * np.pi)),
                           64, 60_000))
    xg, wg = np.polynomial.legendre.leggauss(gl_nodes)
    edges = np.linspace(0.0, upper, n_panels + 1)
    half = 0.5 * (edges[1] - edges[0])

    total = np.zeros_like(v)
    for lo, hi in zip(edges[:-1], edges[1:]):
        u = 0.5 * (lo + hi) + half * xg                       # (gl_nodes,)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_u = np.log(u)
        log_u = np.nan_to_num(log_u, nan=0.0, neginf=0.0)
        amp = wg * np.exp(-u) * u ** (d - 1)                   # (gl_nodes,)
        phase = v[:, None] * u[None, :] + c[:, None] * (u * log_u)[None, :]
        total += half * (np.cos(phase) @ amp)
    return total / (2.0 * np.pi) ** d


def _bell_complete(k, A, s):
    """Complete Bell polynomial in ``(A, psi', psi'', ...)`` for ``d^k/ds^k``."""
    if k == 0:
        return np.ones_like(A)
    p1 = polygamma(1, s)
    if k == 1:
        return A
    if k == 2:
        return A ** 2 + p1
    p2 = polygamma(2, s)
    if k == 3:
        return A ** 3 + 3.0 * A * p1 + p2
    p3 = polygamma(3, s)
    if k == 4:
        return A ** 4 + 6.0 * A ** 2 * p1 + 4.0 * A * p2 + 3.0 * p1 ** 2 + p3
    p4 = polygamma(4, s)
    if k == 5:
        return (A ** 5 + 10.0 * A ** 3 * p1 + 10.0 * A ** 2 * p2
                + 15.0 * A * p1 ** 2 + 5.0 * A * p3 + 10.0 * p1 * p2 + p4)
    raise ValueError("terms beyond k = 5 are not implemented")


def _g_alpha1_asymptotic(v, beta, d, terms=5):
    """Large-``|v|`` series for ``g_{1,d}``; uses ``g(v,b) = g(-v,-b)``."""
    v = np.atleast_1d(np.asarray(v, dtype=np.float64))
    beta = np.broadcast_to(np.asarray(beta, dtype=np.float64), v.shape)
    c = (2.0 / np.pi) * np.where(v < 0.0, -beta, beta)
    x = np.abs(v)
    p = 1.0 - 1j * x
    log_p = np.log(p)

    total = np.zeros(v.shape, dtype=np.complex128)
    for k in range(terms + 1):
        s = d + k
        A = digamma(s) - log_p
        total += ((1j * c) ** k / math.factorial(k)) * gamma_fn(s) * p ** (-s) \
            * _bell_complete(k, A, s)
    return np.real(total) / (2.0 * np.pi) ** d


def _g_alpha1(v, beta, d):
    """``g_{1,d}(v, beta)`` -- exact, valid for every ``v`` and ``beta``."""
    v = np.atleast_1d(np.asarray(v, dtype=np.float64))
    beta = np.broadcast_to(np.asarray(beta, dtype=np.float64), v.shape)
    out = np.empty_like(v)

    symmetric = beta == 0.0
    if np.any(symmetric):
        out[symmetric] = _g_alpha1_symmetric(v[symmetric], d)

    far = (~symmetric) & (np.abs(v) > _A1_ASYMPTOTIC_V)
    if np.any(far):
        out[far] = _g_alpha1_asymptotic(v[far], beta[far], d)

    near = (~symmetric) & (~far)
    if np.any(near):
        out[near] = _g_alpha1_panels(v[near], beta[near], d)
    return out


def g_alpha_d(v, beta, alpha, d, *, quad_limit=200):
    """Projected one dimensional function ``g_{alpha,d}(v, beta)`` of eq. (1).

    ``v`` and ``beta`` are broadcastable arrays; ``beta`` is the (A) skewness of
    the projection (equation (5)).
    """
    v = np.asarray(v, dtype=np.float64)
    beta = np.broadcast_to(np.asarray(beta, dtype=np.float64),
                           v.shape).astype(np.float64)
    if alpha == 1.0:
        return _g_alpha1(v, beta, d).reshape(v.shape)
    return _g_general(v, beta, alpha, d, quad_limit)


# --------------------------------------------------------------------------- #
# Spline surrogate of the kernel (the performance path)
# --------------------------------------------------------------------------- #
class _KernelSpline:
    """Spline of ``g_{alpha,d}(v, beta)`` in ``v`` (and ``beta`` if it varies).

    Nodes are equally spaced in ``arcsinh(v)``, which is linear near the origin
    and logarithmic in the tail -- matching the ``|v|^{-d}`` decay of ``g``.
    Beyond ``v_max`` the closed-form tail is used, so no extrapolation happens.
    """

    #: Kernel nodes are evaluated in one ``quad_vec`` call.  Splitting them into
    #: ``arcsinh(v)`` bands (so that adaptive refinement driven by the stiff
    #: large-``|v|`` nodes stays local) was measured to be ~3x *slower*: the
    #: per-call setup of ``quad_vec`` dominates the subdivision it saves.
    _N_BANDS = 1

    def __init__(self, alpha, d, betas, *, v_max=1.0e3, n_v=None,
                 n_beta=25, beta_tol=1e-9):
        self.alpha = float(alpha)
        self.d = int(d)
        self.v_max = float(v_max)

        betas = np.atleast_1d(np.asarray(betas, dtype=np.float64))
        lo, hi = float(np.min(betas)), float(np.max(betas))
        self._constant_beta = (hi - lo) <= beta_tol
        self._beta_value = 0.5 * (lo + hi)

        s_max = np.arcsinh(self.v_max)
        if self._constant_beta:
            n_v = 701 if n_v is None else n_v
            s_nodes = np.linspace(-s_max, s_max, n_v)
            v_nodes = np.sinh(s_nodes)
            g_nodes = self._kernel_on_nodes(v_nodes, self._beta_value)
            self._spline = CubicSpline(s_nodes, g_nodes, extrapolate=False)
        else:
            n_v = 401 if n_v is None else n_v
            s_nodes = np.linspace(-s_max, s_max, n_v)
            v_nodes = np.sinh(s_nodes)
            # pad the beta range slightly so queries sit inside the grid
            pad = 0.02 * max(hi - lo, 1e-3)
            b_nodes = np.linspace(max(lo - pad, -1.0), min(hi + pad, 1.0),
                                  max(n_beta, 6))
            grid = np.empty((n_v, b_nodes.size))
            for j, b in enumerate(b_nodes):
                grid[:, j] = self._kernel_on_nodes(v_nodes, b)
            self._spline = RectBivariateSpline(s_nodes, b_nodes, grid,
                                               kx=3, ky=3)
            self._b_lo, self._b_hi = b_nodes[0], b_nodes[-1]

    def _kernel_on_nodes(self, v_nodes, beta):
        """Exact kernel at the (monotonically ordered) nodes, band by band."""
        out = np.empty_like(v_nodes)
        edges = np.linspace(0, v_nodes.size, self._N_BANDS + 1).astype(int)
        for lo, hi in zip(edges[:-1], edges[1:]):
            if hi <= lo:
                continue
            chunk = v_nodes[lo:hi]
            out[lo:hi] = g_alpha_d(chunk, np.full_like(chunk, beta),
                                   self.alpha, self.d)
        return out

    def _tail(self, v, beta):
        if self.alpha == 1.0:
            return _g_alpha1_asymptotic(v, beta, self.d)
        # alpha != 1: the h-function representation is cheap and exact here.
        return _g_general(v, np.broadcast_to(beta, v.shape).astype(np.float64),
                          self.alpha, self.d, 200)

    def __call__(self, v, beta):
        v = np.asarray(v, dtype=np.float64)
        beta = np.broadcast_to(np.asarray(beta, dtype=np.float64), v.shape)
        out = np.empty(v.shape, dtype=np.float64)

        inside = np.abs(v) <= self.v_max
        if np.any(inside):
            s = np.arcsinh(v[inside])
            if self._constant_beta:
                out[inside] = self._spline(s)
            else:
                b = np.clip(beta[inside], self._b_lo, self._b_hi)
                out[inside] = self._spline.ev(s, b)
        if not np.all(inside):
            outside = ~inside
            out[outside] = self._tail(v[outside], beta[outside])
        return out


# --------------------------------------------------------------------------- #
# Sphere quadrature and projection parameters
# --------------------------------------------------------------------------- #
def _sphere_directions(d, number_of_points, random_state=None, *, method="mc",
                       antipodal=False):
    """Quadrature nodes ``s_j`` on the unit sphere and the common weight.

    Returns ``(points, weight)`` with ``points`` of shape ``(m, d)`` and
    ``weight`` such that ``\\int_{S^{d-1}} f ds ~= weight * sum_j f(s_j)``.

    ``d == 1`` is exact (``S^0 = {-1, +1}``), ``d == 2`` uses a deterministic
    equi-angular grid (spectrally accurate for the smooth periodic angular
    integral).  For ``d > 2``, ``method`` may be ``"mc"`` or ``"sobol"``.
    The latter maps a scrambled Sobol sequence through independent normal
    quantiles and normalizes it onto the sphere.
    """
    if d == 1:
        return np.array([[1.0], [-1.0]]), 1.0
    if d == 2:
        phi = np.linspace(0.0, 2.0 * np.pi, number_of_points, endpoint=False)
        points = np.column_stack([np.cos(phi), np.sin(phi)])
        weight = 2.0 * np.pi / number_of_points
        return points, weight

    if method not in {"mc", "sobol"}:
        raise ValueError("sphere method must be 'mc' or 'sobol'")

    base_count = (number_of_points + 1) // 2 if antipodal else number_of_points
    if method == "mc":
        rng = np.random.default_rng(random_state)
        points = rng.normal(size=(base_count, d))
    else:
        engine = qmc.Sobol(d=d, scramble=True, seed=random_state)
        u = engine.random(base_count)
        eps = np.finfo(np.float64).eps
        points = norm.ppf(np.clip(u, eps, 1.0 - eps))

    points /= np.linalg.norm(points, axis=1, keepdims=True)
    if antipodal:
        points = np.concatenate([points, -points], axis=0)[:number_of_points]
    area = 2.0 * np.pi ** (d / 2.0) / gamma_fn(d / 2.0)
    weight = area / points.shape[0]
    return points, weight


def _projection_parameters(directions, samples, alpha, mass):
    """``sigma(s)``, ``beta(s)`` and (for alpha == 1) ``mu(s)`` per direction."""
    proj = directions @ samples.T                    # (m, N) = <s_j, V_i>
    abs_proj = np.abs(proj)

    sigma_alpha = mass * np.mean(abs_proj ** alpha, axis=1)
    num = mass * np.mean(np.sign(proj) * abs_proj ** alpha, axis=1)
    sigma = sigma_alpha ** (1.0 / alpha)
    beta = np.where(sigma_alpha > 0.0, num / sigma_alpha, 0.0)
    beta = np.clip(beta, -1.0, 1.0)

    mu = None
    if alpha == 1.0:
        sample_norms = np.linalg.norm(samples, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_unit = np.log(abs_proj / sample_norms[None, :])
        log_unit = np.nan_to_num(log_unit, nan=0.0, posinf=0.0, neginf=0.0)
        mu = -(2.0 / np.pi) * mass * np.mean(proj * log_unit, axis=1)
    return sigma, beta, mu


def _projection_parameters_batched(directions, samples, alpha, mass,
                                   batch_size=10_000):
    """Batched equivalent of :func:`_projection_parameters`.

    The full implementation needs ``len(directions) * len(samples)`` floating
    point values at once.  This version bounds the temporary projection matrix
    to ``len(directions) * batch_size`` while accumulating the same empirical
    moments.
    """
    directions = np.asarray(directions, dtype=np.float64)
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] == 0:
        raise ValueError("samples must be a non-empty two-dimensional array")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    sum_abs_alpha = np.zeros(directions.shape[0], dtype=np.float64)
    sum_signed_abs_alpha = np.zeros_like(sum_abs_alpha)
    sum_mu = np.zeros_like(sum_abs_alpha) if alpha == 1.0 else None

    for start in range(0, samples.shape[0], batch_size):
        block = samples[start:start + batch_size]
        proj = directions @ block.T
        abs_proj = np.abs(proj)
        abs_alpha = abs_proj ** alpha
        sum_abs_alpha += np.sum(abs_alpha, axis=1)
        sum_signed_abs_alpha += np.sum(np.sign(proj) * abs_alpha, axis=1)

        if sum_mu is not None:
            sample_norms = np.linalg.norm(block, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                log_unit = np.log(abs_proj / sample_norms[None, :])
            log_unit = np.nan_to_num(log_unit, nan=0.0, posinf=0.0,
                                     neginf=0.0)
            sum_mu += np.sum(proj * log_unit, axis=1)

    factor = float(mass) / samples.shape[0]
    sigma_alpha = factor * sum_abs_alpha
    numerator = factor * sum_signed_abs_alpha
    sigma = sigma_alpha ** (1.0 / alpha)
    beta = np.divide(numerator, sigma_alpha, out=np.zeros_like(numerator),
                     where=sigma_alpha > 0.0)
    beta = np.clip(beta, -1.0, 1.0)
    mu = None if sum_mu is None else -(2.0 / np.pi) * factor * sum_mu
    return sigma, beta, mu


def _projection_parameters_discrete(directions, positions, weights, alpha):
    """Exact projection parameters for a finite spectral measure."""
    directions = np.asarray(directions, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    proj = directions @ positions.T
    abs_proj = np.abs(proj)
    weighted_abs_alpha = abs_proj ** alpha * weights[None, :]
    sigma_alpha = np.sum(weighted_abs_alpha, axis=1)
    numerator = np.sum(np.sign(proj) * weighted_abs_alpha, axis=1)
    sigma = sigma_alpha ** (1.0 / alpha)
    beta = np.divide(numerator, sigma_alpha, out=np.zeros_like(numerator),
                     where=sigma_alpha > 0.0)
    beta = np.clip(beta, -1.0, 1.0)

    mu = None
    if alpha == 1.0:
        position_norms = np.linalg.norm(positions, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_unit = np.log(abs_proj / position_norms[None, :])
        log_unit = np.nan_to_num(log_unit, nan=0.0, posinf=0.0, neginf=0.0)
        mu = -(2.0 / np.pi) * np.sum(
            proj * log_unit * weights[None, :], axis=1)
    return sigma, beta, mu


def _projection_parameters_isotropic(directions, gamma_scale, alpha):
    """Exact parameters of a rotationally invariant stable law."""
    sigma = np.full(directions.shape[0], float(gamma_scale))
    beta = np.zeros(directions.shape[0], dtype=np.float64)
    mu = np.zeros(directions.shape[0], dtype=np.float64) if alpha == 1.0 else None
    return sigma, beta, mu


def _independent_axis_parameters(sampler, alpha):
    """Return marginal ``(sigma, beta)`` for an axis-supported measure."""
    positions = np.asarray(sampler.positions, dtype=np.float64)
    weights = np.asarray(sampler.weights, dtype=np.float64)
    nonzero = np.abs(positions) > 0.0
    if np.any(np.sum(nonzero, axis=1) > 1):
        return None
    sigma_alpha = np.sum(
        weights[:, None] * np.abs(positions) ** alpha, axis=0)
    if np.any(sigma_alpha <= 0.0):
        return None
    numerator = np.sum(
        weights[:, None] * np.sign(positions) * np.abs(positions) ** alpha,
        axis=0)
    sigma = sigma_alpha ** (1.0 / alpha)
    beta = np.divide(numerator, sigma_alpha, out=np.zeros_like(numerator),
                     where=sigma_alpha > 0.0)
    return sigma, np.clip(beta, -1.0, 1.0)


def _projection_parameters_elliptical(directions, scatter, mass, alpha):
    """Exact parameters for ``(s.T @ scatter @ s) ** (alpha / 2)``."""
    scatter = np.asarray(scatter, dtype=np.float64)
    quadratic = np.einsum("ij,jk,ik->i", directions, scatter, directions)
    sigma = float(mass) ** (1.0 / alpha) * np.sqrt(
        np.maximum(quadratic, 0.0))
    beta = np.zeros(directions.shape[0], dtype=np.float64)
    mu = np.zeros(directions.shape[0], dtype=np.float64) if alpha == 1.0 else None
    return sigma, beta, mu


def _isotropic_radial_pdf(radius, alpha, d, gamma_scale, *, gl_nodes=24,
                          damping_cutoff=60.0):
    """Isotropic density from its radial Fourier--Bessel transform."""
    radius = np.atleast_1d(np.asarray(radius, dtype=np.float64))
    unique_radius, inverse = np.unique(radius, return_inverse=True)
    out = np.empty_like(unique_radius)
    at_origin = unique_radius <= 1e-12
    if np.any(at_origin):
        sphere_area = 2.0 * np.pi ** (d / 2.0) / gamma_fn(d / 2.0)
        out[at_origin] = (
            sphere_area * gamma_fn(d / alpha)
            / (alpha * (2.0 * np.pi) ** d * gamma_scale ** d)
        )

    active = ~at_origin
    if np.any(active):
        r = unique_radius[active]
        upper = damping_cutoff ** (1.0 / alpha) / gamma_scale
        max_phase = max(float(np.max(r)) * upper, 1.0)
        panels = max(64, int(np.ceil(2.0 * max_phase / (2.0 * np.pi))))
        nodes, weights = np.polynomial.legendre.leggauss(gl_nodes)
        edges = np.linspace(0.0, upper, panels + 1)
        order = d / 2.0 - 1.0
        half = 0.5 * (edges[1:] - edges[:-1])
        centres = 0.5 * (edges[1:] + edges[:-1])
        rho = (centres[:, None] + half[:, None] * nodes[None, :]).ravel()
        quadrature_weights = (half[:, None] * weights[None, :]).ravel()
        amplitude = (quadrature_weights * rho ** (d / 2.0)
                     * np.exp(-(gamma_scale * rho) ** alpha))
        total = np.empty_like(r)
        for start in range(0, r.size, 128):
            block = r[start:start + 128]
            total[start:start + 128] = jv(
                order, block[:, None] * rho[None, :]) @ amplitude
        out[active] = ((2.0 * np.pi) ** (-d / 2.0)
                       * r ** (1.0 - d / 2.0) * total)

    np.maximum(out, 0.0, out=out)
    return out[inverse]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
class MultivariateStableDensity:
    """Reusable multivariate alpha-stable density.

    The spectral-measure sampling, sphere quadrature and projection parameters
    are computed once here, so evaluating many points (or many grids) costs only
    the kernel evaluation.

    Parameters
    ----------
    alpha : float
        Stability index in ``(0, 2)``.
    spectral_measure_sampler : BaseSpectralMeasureSampler
        Sampler describing the spectral measure ``Gamma``.
    shift : array_like, optional
        Location vector ``nu``.
    number_of_spectral_samples : int, optional
        Monte-Carlo samples used for ``sigma``/``beta``/``mu``.
    number_of_sphere_points : int, optional
        Sphere quadrature nodes.  Defaults to 360 for ``d == 2`` (the angular
        integrand is smooth and periodic, so this is already converged) and
        4096 otherwise (a power of two preserves Sobol balance properties).
    sphere_method : {"mc", "sobol"}, optional
        Higher-dimensional sphere-node generator.  Ignored in dimensions one
        and two.  Defaults to scrambled Sobol, which has substantially lower
        error and seed variance in the bundled convergence benchmark.
    antipodal : bool, optional
        Pair each higher-dimensional node with its negative.
    random_state : None | int | np.random.Generator, optional
        Seeds the spectral samples and, for ``d > 2``, the sphere nodes.
    exact : bool, optional
        Evaluate the kernel directly instead of through the spline surrogate.
        Slower; used as the reference in tests.
    projection_batch_size : int, optional
        Spectral samples projected at once for non-discrete measures.  This
        bounds setup memory independently of the total sample count.
    """

    def __init__(self, alpha, spectral_measure_sampler: BaseSpectralMeasureSampler,
                 shift=0.0, number_of_spectral_samples=200_000,
                 number_of_sphere_points=None, random_state=None,
                 exact=False, projection_batch_size=1_000,
                 sphere_method="sobol", antipodal=False):
        if not 0.0 < alpha <= 2.0:
            raise ValueError(f"alpha must lie in (0, 2], got {alpha}")
        alpha = float(alpha)
        if alpha == 2.0:
            raise ValueError(
                "alpha == 2 is the Gaussian limit; the projection kernel is "
                "singular there (tan(pi alpha / 2) diverges).  Use a Gaussian "
                "density with covariance 2 * mass * E[V V^T] instead."
            )

        d = spectral_measure_sampler.dimensions()
        self.alpha = alpha
        self.dimensions = d
        self.sampler = spectral_measure_sampler
        self._independent_params = (
            _independent_axis_parameters(spectral_measure_sampler, alpha)
            if isinstance(spectral_measure_sampler, DiscreteSampler) else None
        )
        self.exact = bool(exact)
        self.shift = np.broadcast_to(np.asarray(shift, dtype=np.float64), (d,))

        if number_of_sphere_points is None:
            number_of_sphere_points = 360 if d == 2 else 4096
        self.number_of_sphere_points = int(number_of_sphere_points)

        self.directions, self.weight = _sphere_directions(
            d, self.number_of_sphere_points, random_state,
            method=sphere_method, antipodal=antipodal)
        if isinstance(spectral_measure_sampler, IsotropicSampler):
            self.sigma, self.beta, self.mu = _projection_parameters_isotropic(
                self.directions, spectral_measure_sampler.gamma, alpha)
        elif isinstance(spectral_measure_sampler, EllipticSampler):
            self.sigma, self.beta, self.mu = _projection_parameters_elliptical(
                self.directions, spectral_measure_sampler.sigma,
                spectral_measure_sampler.mass(), alpha)
        elif isinstance(spectral_measure_sampler, DiscreteSampler):
            self.sigma, self.beta, self.mu = _projection_parameters_discrete(
                self.directions, spectral_measure_sampler.positions,
                spectral_measure_sampler.weights, alpha)
        else:
            samples = spectral_measure_sampler.sample(number_of_spectral_samples,
                                                      random_state)
            self.sigma, self.beta, self.mu = _projection_parameters_batched(
                self.directions, samples, alpha, spectral_measure_sampler.mass(),
                batch_size=projection_batch_size)

        self._valid = self.sigma > 0.0
        self._inv_sigma_d = np.zeros_like(self.sigma)
        self._inv_sigma_d[self._valid] = self.sigma[self._valid] ** (-d)

        self._kernel = None
        if not self.exact and not isinstance(self.sampler, IsotropicSampler):
            self._kernel = _KernelSpline(alpha, d, self.beta[self._valid])

    # -- internals ------------------------------------------------------ #
    def _v_of(self, dot):
        """Projection argument ``v`` for a batch of ``<x - nu, s>`` values."""
        valid = self._valid
        sigma = self.sigma
        v = np.zeros_like(dot)
        if self.alpha == 1.0:
            v[:, valid] = (
                dot[:, valid] - self.mu[valid]
                - (2.0 / np.pi) * self.beta[valid] * sigma[valid]
                * np.log(sigma[valid])
            ) / sigma[valid]
        else:
            v[:, valid] = dot[:, valid] / sigma[valid]
        return v

    def _kernel_values(self, v_valid, beta_valid):
        if self._kernel is not None:
            return self._kernel(v_valid, beta_valid)
        return g_alpha_d(v_valid, beta_valid, self.alpha, self.dimensions)

    # -- public --------------------------------------------------------- #
    def pdf(self, x, *, chunk_size=200_000):
        """Density at ``x`` of shape ``(d,)`` or ``(n, d)``."""
        x = np.asarray(x, dtype=np.float64)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.dimensions:
            raise ValueError(
                f"x has dimension {x.shape[-1]} but the spectral measure is "
                f"{self.dimensions}-dimensional"
            )

        if isinstance(self.sampler, IsotropicSampler):
            densities = _isotropic_radial_pdf(
                np.linalg.norm(x - self.shift, axis=1), self.alpha,
                self.dimensions, self.sampler.gamma)
            return densities[0] if single else densities

        if isinstance(self.sampler, EllipticSampler) and self.sampler.mass() > 0.0:
            transform = (self.sampler.mass() ** (1.0 / self.alpha)
                         * np.linalg.cholesky(self.sampler.sigma))
            radial_coordinates = np.linalg.solve(
                transform, (x - self.shift).T).T
            densities = _isotropic_radial_pdf(
                np.linalg.norm(radial_coordinates, axis=1), self.alpha,
                self.dimensions, gamma_scale=1.0)
            densities /= abs(np.linalg.det(transform))
            return densities[0] if single else densities

        if self._independent_params is not None:
            # Import lazily to avoid the public wrapper's import cycle while
            # this module is initialized.  Axis-supported spectral measures
            # factorize exactly into the package's own univariate S1 laws.
            from aub_htp import alpha_stable
            marginal_sigma, marginal_beta = self._independent_params
            densities = np.ones(x.shape[0], dtype=np.float64)
            for axis in range(self.dimensions):
                coordinates, inverse = np.unique(
                    x[:, axis], return_inverse=True)
                values = alpha_stable.pdf(
                    coordinates, self.alpha, marginal_beta[axis],
                    loc=self.shift[axis], scale=marginal_sigma[axis])
                densities *= values[inverse]
            return densities[0] if single else densities

        valid = self._valid
        beta_valid = self.beta[valid]
        scale = self._inv_sigma_d[valid]
        n_valid = int(valid.sum())
        if n_valid == 0:
            out = np.zeros(x.shape[0])
            return out[0] if single else out

        # Process points in chunks so the (chunk x directions) kernel array
        # stays bounded regardless of how many points were requested.
        rows = max(1, int(chunk_size // max(n_valid, 1)))
        densities = np.empty(x.shape[0])
        centred = x - self.shift
        for lo in range(0, x.shape[0], rows):
            block = centred[lo:lo + rows]
            dot = block @ self.directions.T                 # (rows, m)
            v = self._v_of(dot)[:, valid]                   # (rows, m_valid)
            g = self._kernel_values(
                v.ravel(), np.broadcast_to(beta_valid, v.shape).ravel()
            ).reshape(v.shape)
            densities[lo:lo + rows] = self.weight * (g * scale).sum(axis=1)

        np.maximum(densities, 0.0, out=densities)
        return densities[0] if single else densities

    __call__ = pdf


def get_pdf_model(
    alpha: float,
    spectral_measure_sampler: BaseSpectralMeasureSampler,
    shift: npt.ArrayLike = 0.0,
    number_of_spectral_samples: int = 200_000,
    number_of_sphere_points: int | None = None,
    random_state=None,
    exact: bool = False,
    sphere_method: str = "sobol",
    antipodal: bool = False,
) -> MultivariateStableDensity:
    """Return a cached density model, constructing it on a cache miss."""
    key = (
        float(alpha),
        _measure_key(spectral_measure_sampler),
        _array_key(np.asarray(shift, dtype=np.float64)),
        number_of_spectral_samples,
        number_of_sphere_points,
        _random_state_key(random_state),
        bool(exact),
        sphere_method,
        bool(antipodal),
    )
    try:
        model = _pdf_model_cache.pop(key)
    except KeyError:
        model = MultivariateStableDensity(
            alpha,
            spectral_measure_sampler,
            shift=shift,
            number_of_spectral_samples=number_of_spectral_samples,
            number_of_sphere_points=number_of_sphere_points,
            random_state=random_state,
            exact=exact,
            sphere_method=sphere_method,
            antipodal=antipodal,
        )
    _pdf_model_cache[key] = model
    if len(_pdf_model_cache) > _PDF_MODEL_CACHE_MAXSIZE:
        _pdf_model_cache.popitem(last=False)
    return model


def multivariate_alpha_stable_pdf(
    x,
    alpha,
    spectral_measure_sampler: BaseSpectralMeasureSampler,
    shift=0.0,
    number_of_spectral_samples=200_000,
    number_of_sphere_points=None,
    random_state=None,
    exact=False,
    sphere_method="sobol",
    antipodal=False,
):
    """Numerical multivariate alpha-stable density.

    Thin cached wrapper over :class:`MultivariateStableDensity` for one-shot
    use. Construct the class directly when explicit control over a model's
    lifetime is preferred.

    Parameters
    ----------
    x : array_like
        Point(s) at which to evaluate the density: shape ``(d,)`` or ``(n, d)``.
    alpha : float
        Stability index in ``(0, 2)``.
    spectral_measure_sampler : BaseSpectralMeasureSampler
        Sampler describing the spectral measure ``Gamma``.
    shift : array_like, optional
        Location vector ``nu``.
    number_of_spectral_samples : int, optional
        Monte-Carlo samples used to estimate the projection parameters.
    number_of_sphere_points : int, optional
        Sphere quadrature nodes.
    random_state : None | int | np.random.Generator, optional
        Seeds the spectral samples and, for ``d > 2``, the sphere nodes.
    exact : bool, optional
        Bypass the spline surrogate and evaluate the kernel directly.
    sphere_method : {"mc", "sobol"}, optional
        Higher-dimensional sphere-node generator.
    antipodal : bool, optional
        Pair higher-dimensional sphere nodes with their negatives.

    Returns
    -------
    density : float or np.ndarray
        Scalar for a single point, otherwise shape ``(n,)``.
    """
    model = get_pdf_model(
        alpha,
        spectral_measure_sampler,
        shift=shift,
        number_of_spectral_samples=number_of_spectral_samples,
        number_of_sphere_points=number_of_sphere_points,
        random_state=random_state,
        exact=exact,
        sphere_method=sphere_method,
        antipodal=antipodal,
    )
    return model.pdf(x)
