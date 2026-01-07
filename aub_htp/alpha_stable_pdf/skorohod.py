import numpy as np
from scipy.special import gamma
from scipy.special import factorial
from math import log
from scipy.integrate import quad
from functools import lru_cache


def skorohod_formula_1(x, alpha, beta, N=170):
    """
    Skorohod's Formula 1: series for 0 < alpha < 1 (tails).

    - Uses truncated series with N terms.
    - Valid only when 0 < alpha < 1 and |beta| <= 1.
    - Works on x > 0 for right tail; use |x| and flipped beta for left tail.
    Returns vector of pdf approximations at x.
    """

    if not (0 < alpha < 1 and 0 <= abs(beta) <= 1):
        print("Conditions not met for using Skorohod's first formula.")
        return None

    n = np.arange(1, N + 1)[:, np.newaxis]  # shape (N, 1)

    coeffs = ((-1) ** (n - 1)) * gamma(n * alpha + 1) / factorial(n)
    factor = (1 + beta ** 2 * (np.tan(np.pi * alpha / 2)) ** 2) ** (n / 2)
    angle = n * (np.pi / 2 * alpha + np.arctan(beta * np.tan(np.pi * alpha / 2)))
    terms = coeffs * factor * np.sin(angle)  # shape (N, 1)

    x_powers = x[np.newaxis, :] ** (-alpha * n)  # shape (N, len(x))
    sum_terms = np.sum(terms * x_powers, axis=0)  # shape (len(x),)

    return sum_terms / (np.pi * x)


@lru_cache(maxsize=128)
def skorohod_formula_2_bk(beta, k):
    """
    Helper for Skorohod's Formula 2 (alpha = 1): compute b_k(beta).

    - Integrates e^{-v} v^k * Im[ 1j + beta*1j - (2*beta/π) ln v ] dv over v∈[0, ∞).
    - Caches results by (beta, k) to speed up repeated calls.
    """

    def integrand(v):
        complex_term = 1j + beta * 1j - (2 * beta / np.pi) * np.log(v)
        return np.exp(-v) * np.power(v, k) * complex_term.imag

    result, _ = quad(integrand, 0, 10000)
    return result


def skorohod_formula_2(x, beta, N=10):
    """
    Skorohod's Formula 2: alpha = 1 case (Cauchy-like).

    - Adjust x by log term from skewness.
    - Use truncated inverse-power series with coefficients b_k(beta).
    - Valid for x ≠ 0; typically used on x > 0 and mirrored for x < 0.
    """
    x = np.asarray(x, dtype=np.float64)

    adjusted_x = x + beta ** 2 * (2 / np.pi) * np.log(x)

    ks = np.arange(1, N + 1)[:, np.newaxis]  # shape (N, 1)
    bks = np.array([skorohod_formula_2_bk(beta, k) for k in range(1, N + 1)])[:, np.newaxis]  # shape (N, 1)

    x_powers = adjusted_x[np.newaxis, :] ** (-ks)  # shape (N, len(x))

    sum_terms = np.sum(bks * x_powers, axis=0)  # shape (len(x),)

    return 1 / (np.pi * adjusted_x) * sum_terms


def skorohod_formula_3_an(alpha, beta, n):
    """
    Coefficient a_n(alpha, beta) for Skorohod's Formula 3 (1 < alpha < 2 tails).

    - Returns array for vector n.
    - a_n = (-1)^{n-1} (1 + β^2 tan^2(πα/2))^{n/2} sin(n(πα/2 + atan(β tan(πα/2)))) Γ(nα+1)/n!
    """
    n = np.asarray(n, dtype=np.float64)

    term1 = (-1) ** (n - 1)
    term2 = (1 + (beta ** 2) * (np.tan(np.pi * alpha / 2)) ** 2) ** (n / 2)
    term3 = np.sin(n * (np.pi * alpha / 2 + np.arctan(beta * np.tan(np.pi * alpha / 2))))

    exact = gamma(n * alpha + 1) / factorial(n)

    return term1 * term2 * term3 * exact


def skorohod_formula_3(x, alpha, beta, N=20):
    """
    Skorohod's Formula 3: series for 1 < alpha < 2 (tails).

    - Uses a_n(alpha, beta) with N terms.
    - Typically applied to |x| large.
    """
    x = np.asarray(x, dtype=np.float64)

    n_vals = np.arange(1, N + 1)
    an_vals = skorohod_formula_3_an(alpha, beta, n_vals)[:, np.newaxis]

    x_powers = x[np.newaxis, :] ** (-alpha * n_vals[:, np.newaxis])

    sum_terms = np.sum(an_vals * x_powers, axis=0)

    return (1 / (np.pi * x)) * sum_terms


def skorohod_formula_4_A(alpha):
    """
    Asymptotic prefactor A(alpha) for 0 < alpha < 1 near zero (β = ±1 edges).
    """
    return ((alpha ** (1 / (2 - 2 * alpha))) *
            np.cos(np.pi * alpha / 2) ** (-1 / (2 - 2 * alpha))) / (
        np.sqrt(2 * np.pi * (1 - alpha)))


def skorohod_formula_4_B(alpha):
    """
    Asymptotic exponent scale B(alpha) for 0 < alpha < 1 near zero.
    """
    return ((1 - alpha) *
            (alpha ** (alpha / (1 - alpha))) *
            np.cos(np.pi * alpha / 2) ** (-1 / (1 - alpha)))


def skorohod_formula_4_Lambda(alpha):
    """
    Lambda(alpha) = alpha / (1 - alpha) for 0 < alpha < 1.
    """
    return alpha / (1 - alpha)


def skorohod_formula_4(x, alpha):
    """
    Skorohod's Formula 4: small-|x| asymptotic for 0 < alpha < 1.

    - Used when x→0 for extreme skew β=±1 handling.
    - Returns the core exponential form A x^{-1-λ/2} exp(-B x^{-λ}).
    """
    x = np.asarray(x, dtype=np.float64)

    a = skorohod_formula_4_A(alpha)
    b = skorohod_formula_4_B(alpha)
    lambda_ = skorohod_formula_4_Lambda(alpha)

    core = a * x ** (-1 - lambda_ / 2) * np.exp(-b * x ** (-lambda_))
    return core  # or `core * (1 + x ** (lambda_ / 2))` if you want to include the optional term


def skorohod_formula_5(x):
    """
    Skorohod's Formula 5: alpha = 1, near zero correction (heuristic).

    - Empirical correction term to stabilize behavior around 0.
    - Not a strict series; acts as a patch for numerical issues.
    """
    x = np.asarray(x, dtype=np.float64)
    part1 = 1 / (np.pi * np.sqrt(np.e))
    part2 = np.exp((-np.pi / 4) * x - (2 / (np.pi * np.e)) * np.exp((-np.pi / 2) * x))
    correction = 1 + np.exp((x * np.pi / 4) * 0.56)
    return part1 * part2 * correction


def skorohod_formula_6_A_prime(alpha):
    """
    Asymptotic prefactor A'(alpha) for alpha > 1 near zero (β = ±1 edges).
    """
    term1 = alpha ** (-1 / (2 * (alpha - 1)))
    term2 = np.abs(np.cos(np.pi * alpha / 2)) ** (1 / (2 * (alpha - 1)))
    term3 = np.sqrt(2 * np.pi * (alpha - 1))
    return (term1 * term2) / term3


def skorohod_formula_6_B_prime(alpha):
    """
    Asymptotic exponent scale B'(alpha) for alpha > 1 near zero.
    """
    term1 = alpha - 1
    term2 = alpha ** (-alpha / (alpha - 1))
    term3 = np.abs(np.cos(np.pi * alpha / 2)) ** (1 / (alpha - 1))
    return term1 * term2 * term3


def skorohod_formula_6_lambda_prime(alpha):
    """
    Lambda'(alpha) = alpha / (alpha - 1) for alpha > 1.
    """
    return alpha / (alpha - 1)


def skorohod_formula_6(x, alpha):
    """
    Skorohod's Formula 6: small-|x| asymptotic for alpha > 1.

    - Used when x→0 for extreme skew β=±1 handling.
    - Returns A' x^{-1+λ'/2} exp(-B' x^{λ'}).
    """
    x = np.asarray(x, dtype=np.float64)

    a_prime = skorohod_formula_6_A_prime(alpha)
    b_prime = skorohod_formula_6_B_prime(alpha)
    lambda_p = skorohod_formula_6_lambda_prime(alpha)

    core = a_prime * x ** (-1 + lambda_p / 2) * np.exp(-b_prime * x ** lambda_p)
    return core


def generate_pdf_skorohod_vectorized(x, alpha, beta, epsilon=1e-6):
    """
    Vectorized dispatcher that combines Skorohod formulas into a stable pdf
    approximation across parameter regimes.

    Inputs
    - x: array of evaluation points (after normalization if needed).
    - alpha ∈ (0, 2], beta ∈ [-1, 1].
    - epsilon: small threshold around zero to switch to near-zero asymptotics.

    Logic
    - alpha < 1:
        - beta =  1: use Formula 1 on x>ε; Formula 4 near zero on (0, ε].
        - beta = -1: mirror to left tail; use |x| and flip beta where needed.
        - else: use Formula 1 on both sides, flipping beta for x<=0.
    - alpha = 1:
        - beta ∈ [-1,1]:
            - main: Formula 2 on each side with |x| as needed.
            - near zero: Formula 5 to stabilize behavior.
    - alpha > 1:
        - beta =  1: Formula 3 on x>0; Formula 6 near/below zero.
        - beta = -1: swap sides; Formula 6 on x>0; Formula 3 on x<=0 with flip.
        - else: Formula 3 on both sides, flipping beta for x<0.

    Returns
    - result: array of pdf approximations, same shape as x.
    """
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x, dtype=np.float64)

    # Validate scalar parameters
    if alpha <= 0 or alpha > 2 or abs(beta) > 1:
        raise ValueError("Invalid parameters: 0 < alpha ≠ 1 < 2 and |beta| ≤ 1")

    if alpha < 1:
        if beta == 1:
            mask_1 = x > epsilon
            mask_2 = (x > 0) & (x <= epsilon)
            result[mask_1] = skorohod_formula_1(x[mask_1], alpha, beta)
            result[mask_2] = skorohod_formula_4(x[mask_2], alpha)

        elif beta == -1:
            mask_3 = x <= -epsilon
            mask_4 = (-epsilon < x) & (x < 0)
            print(mask_3)
            print(mask_4)
            result[mask_3] = skorohod_formula_1(np.abs(x[mask_3]), alpha, -beta)
            result[mask_4] = skorohod_formula_4(np.abs(x[mask_4]), alpha)

        else:  # -1 < beta < 1
            mask_5 = x > 0
            mask_6 = x <= 0
            result[mask_5] = skorohod_formula_1(x[mask_5], alpha, beta)
            result[mask_6] = skorohod_formula_1(np.abs(x[mask_6]), alpha, -beta)

    elif alpha == 1:
        if beta == 1:
            mask_7 = x > 0
            mask_8 = x <= 0
            result[mask_7] = skorohod_formula_2(x[mask_7], beta)
            result[mask_8] = skorohod_formula_5(np.abs(x[mask_8]))

        elif beta == -1:
            mask_9 = x < 0
            mask_10 = x >= 0
            result[mask_9] = skorohod_formula_2(np.abs(x[mask_9]), beta)
            result[mask_10] = skorohod_formula_5(x[mask_10])

        else:
            mask_11 = x > 0
            mask_12 = x <= 0
            result[mask_11] = skorohod_formula_2(x[mask_11], beta)
            result[mask_12] = skorohod_formula_2(np.abs(x[mask_12]), beta)

    elif alpha > 1:
        if beta == 1:
            mask_13 = x > 0
            mask_14 = x <= 0
            result[mask_13] = skorohod_formula_3(x[mask_13], alpha, beta)
            result[mask_14] = skorohod_formula_6(np.abs(x[mask_14]), alpha)

        elif beta == -1:
            mask_15 = x > 0
            mask_16 = x <= 0
            result[mask_15] = skorohod_formula_6(x[mask_15], alpha)
            result[mask_16] = skorohod_formula_3(np.abs(x[mask_16]), alpha, -beta)

        else:
            mask_17 = x < 0
            mask_18 = x >= 0
            result[mask_17] = skorohod_formula_3(np.abs(x[mask_17]), alpha, -beta)
            result[mask_18] = skorohod_formula_3(x[mask_18], alpha, beta)

    return result
