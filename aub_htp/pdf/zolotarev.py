import numpy as np
from scipy.integrate import quad_vec, quad

def theta0_stable(alpha, beta):
    """
    Zolotarev θ0 pivot for stable laws.
    - For α ≠ 1: θ0 = arctan(β tan(πα/2)) / α
    - For α = 1: special-case pivot (here set to π/2)
    """
    return np.arctan(beta * np.tan(np.pi * alpha / 2)) / alpha if alpha != 1 else np.pi / 2

def reflect_if_negative(x, beta, theta0):
    """
    Enforce x ≥ 0 by symmetry.
    - If x < 0, reflect: x -> -x, β -> -β, θ0 -> -θ0, and mark flipped=True.
    - Else return inputs unchanged, flipped=False.
    """
    if x < 0:
        return -x, -beta, -theta0, True
    return x, beta, theta0, False

def calculate_V(alpha, beta, theta0):
    """
    Build V(θ) kernel for Zolotarev integral forms.
    - α ≠ 1: use Type-B (Zolotarev) representation components.
      V(θ) = [cos(αθ0)]^{1/(α-1)} * [cos(αθ0+(α-1)θ)/cos θ] *
             [cos θ / sin(α(θ+θ0))]^{α/(α-1)}
    - α = 1: use the α=1 kernel with exp term.
    Returns a function V(θ).
    """
    if alpha != 1:
        x = (np.cos(alpha * theta0)) ** (1 / (alpha - 1))
        y = lambda theta: np.cos((alpha * theta0) + (alpha - 1) * theta) / np.cos(theta)
        z = lambda theta: ((np.cos(theta)) / (np.sin(alpha * (theta + theta0)))) ** (alpha / (alpha - 1))
        return lambda theta: x * y(theta) * z(theta)
    else:
        return lambda theta: ((2 / np.pi) *
                              ((np.pi / 2 + beta * theta) / np.cos(theta)) *
                              np.exp((np.pi / 2 + beta * theta) * np.tan(theta) / beta))

def generate_pdf_one_point(x, alpha, beta):
    """
    Zolotarev one-point pdf evaluator via 1D quadrature.
    - α ≠ 1: uses Type-B integral on θ ∈ [-θ0, π/2]
      j is the scale Jacobian; integrand uses V(θ) * exp(-x^{α/(α-1)} V(θ)).
    - α = 1, β ≠ 0: uses α=1 kernel on θ ∈ [-π/2, π/2]
      with exponential tilt exp(-π x V(θ) / (2β)).
    Returns (value, estimated error) from quad multiplied by Jacobian.
    """
    if alpha != 1:
        theta0 = theta0_stable(alpha, beta)
        x, beta, theta0, flipped = reflect_if_negative(x, beta, theta0)
        V = calculate_V(alpha, beta, theta0)
        j = (alpha * (x ** (1 / (alpha - 1)))) / (np.pi * abs(alpha - 1))
        exponent = lambda theta: -((x) ** (alpha / (alpha - 1))) * V(theta)
        integrand = lambda theta: V(theta) * np.exp(exponent(theta))
        integral = quad(integrand, -theta0, np.pi / 2, epsabs=1e-8)
        return j * integral[0], j * integral[1]
    elif beta != 0:
        # α = 1 branch with skew β ≠ 0
        theta0 = np.pi / 2
        V = calculate_V(1, beta, theta0)
        j = np.exp(-np.pi * x / (2 * beta)) / (2 * abs(beta))
        integrand = lambda theta: V(theta) * np.exp(-np.pi * x * V(theta) / (2 * beta))
        integral = quad(integrand, -np.pi/2, np.pi/2, epsabs=1e-8)
        return j * integral[0], j * integral[1]
    # Note: α=1, β=0 (Cauchy) not handled here

def generate_pdf_one_point_around_zero(X, alpha, beta):
    """
    Direct CF-based integral for pdf near zero using quad_vec.
    f(x) = (1/π) ∫_0^∞ e^{-t^α} cos(x t − β tan(πα/2) t^α) dt  (S1-style)
    - Vectorized over X using quad_vec.
    - High accuracy tolerances for stability around x≈0.
    """
    X = np.asarray(X, dtype=np.float64)
    tan_part = beta * np.tan(np.pi * alpha / 2)

    def integrand(t, x):
        return np.exp(-t**alpha) * np.cos(x * t - tan_part * t**alpha)

    val, _ = quad_vec(integrand, 0, np.inf, args=(X,), epsabs=1e-12, epsrel=1e-12, limit=100)
    return val / np.pi

def generate_pdf_zolotarev_1(X, alpha, beta):
    """
    Batch wrapper over generate_pdf_one_point using Zolotarev integral.
    - Evaluates at each x in X and collects only the value (drop error).
    """
    pdf_values = np.array([generate_pdf_one_point(x, alpha, beta) for x in X])[:, 0]
    return pdf_values

def generate_pdf_NolanS1(X, alpha, beta):
    """
    CF-integral pdf in Nolan S1 parameterization around zero.
    - Uses generate_pdf_one_point_around_zero for vector X.
    """
    pdf_values = generate_pdf_one_point_around_zero(X, alpha, beta)
    return pdf_values
