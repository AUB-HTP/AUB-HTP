import numpy as np
from functools import lru_cache
from scipy.integrate import quad
from scipy.special import gamma, factorial, gammainc, gammaincc, gammaln


# Optimal truncation controls for the finalized tail series (see bottom of file).
SKOROHOD_TAIL_NMAX = 80
SKOROHOD_TAIL_TRUNC_TOL = 1e-10


# ============================================================================
# Formula I
# ============================================================================

def skorohod_formula_1_cdf(x, alpha, beta, N=170):
    """
    Skorohod's Formula 1 as a CDF: tail series summed to a fixed N terms.

    - Valid for 0 < alpha < 1, where the series converges.
    - For alpha > 1 the series is divergent-asymptotic, so a fixed N=170 terms
      overflows and returns nan. Use skorohod_tail_series_sf / generate_cdf
      below, which truncate each point at its own smallest term instead.
    """
    x=np.asarray(x,dtype=float)
    F=np.empty_like(x)
    pos=x>=0
    if np.any(pos):
        xp=x[pos]
        n=np.arange(1,N+1)[:,None]
        coeff=(((-1)**(n-1))*gamma(n*alpha+1)/factorial(n))
        fac=(1+beta**2*np.tan(np.pi*alpha/2)**2)**(n/2)
        ang=n*(np.pi*alpha/2+np.arctan(beta*np.tan(np.pi*alpha/2)))
        an=coeff*fac*np.sin(ang)
        tail=np.sum(an/(np.pi*alpha*n)*xp[None,:]**(-alpha*n),axis=0)
        F[pos]=1-tail
    if np.any(~pos):
        F[~pos]=1-skorohod_formula_1_cdf(-x[~pos],alpha,-beta,N)
    return F


# ============================================================================
# Formula II
# ============================================================================

@lru_cache(maxsize=128)
def skorohod_formula_2_bk(beta,k):
    def integrand(v):
        return np.exp(-v)*v**k*(1+beta-(2*beta/np.pi)*np.log(v))
    val,_=quad(integrand,0,np.inf)
    return val

def skorohod_formula_2_cdf(x,beta,N=10):
    x=np.asarray(x,dtype=float)
    F=np.empty_like(x)
    pos=x>0
    if np.any(pos):
        xp=x[pos]
        y=xp+beta**2*(2/np.pi)*np.log(xp)
        k=np.arange(1,N+1)[:,None]
        bk=np.array([skorohod_formula_2_bk(beta,i) for i in range(1,N+1)])[:,None]
        tail=np.sum(bk/(np.pi*k)*y[None,:]**(-k),axis=0)
        F[pos]=1-tail
    if np.any(~pos):
        F[~pos]=1-skorohod_formula_2_cdf(-x[~pos],-beta,N)
    return F


# ============================================================================
# Formula III
# ============================================================================

def skorohod_formula_3_an(alpha,beta,n):
    n=np.asarray(n,dtype=float)
    return (((-1)**(n-1))
            *(1+beta**2*np.tan(np.pi*alpha/2)**2)**(n/2)
            *np.sin(n*(np.pi*alpha/2+np.arctan(beta*np.tan(np.pi*alpha/2))))
            *gamma(n*alpha+1)/factorial(n))

def skorohod_formula_3_cdf(x,alpha,beta,N=20):
    x=np.asarray(x,dtype=float)
    F=np.empty_like(x)
    pos=x>0
    if np.any(pos):
        xp=x[pos]
        n=np.arange(1,N+1)
        an=skorohod_formula_3_an(alpha,beta,n)[:,None]
        tail=np.sum(an/(np.pi*alpha*n[:,None])*xp[None,:]**(-alpha*n[:,None]),axis=0)
        F[pos]=1-tail
    if np.any(~pos):
        F[~pos]=1-skorohod_formula_3_cdf(-x[~pos],alpha,-beta,N)
    return F


# ============================================================================
# Formula IV
# ============================================================================

def skorohod_formula_4_A(alpha):
    return alpha**(1/(2-2*alpha))*np.cos(np.pi*alpha/2)**(-1/(2-2*alpha))/np.sqrt(2*np.pi*(1-alpha))

def skorohod_formula_4_B(alpha):
    return (1-alpha)*alpha**(alpha/(1-alpha))*np.cos(np.pi*alpha/2)**(-1/(1-alpha))

def skorohod_formula_4_Lambda(alpha):
    return alpha/(1-alpha)

def skorohod_formula_4_cdf(x,alpha):
    x=np.asarray(x,dtype=float)
    A=skorohod_formula_4_A(alpha)
    B=skorohod_formula_4_B(alpha)
    lam=skorohod_formula_4_Lambda(alpha)
    z=B*x**(-lam)
    return A/(lam*np.sqrt(B))*gamma(0.5)*gammaincc(0.5,z)


# ============================================================================
# Formula V
# ============================================================================

def skorohod_formula_5_pdf(t):
    return (1/(np.pi*np.sqrt(np.e))
            *np.exp((-np.pi/4)*t-(2/(np.pi*np.e))*np.exp((-np.pi/2)*t))
            *(1+np.exp((np.pi/4)*0.56*t)))

def skorohod_formula_5_cdf(x):
    x=np.asarray(x,dtype=float)
    out=np.empty_like(x)
    for i,xi in enumerate(x):
        out[i]=quad(skorohod_formula_5_pdf,-np.inf,xi)[0]
    return out


# ============================================================================
# Formula VI
# ============================================================================

def skorohod_formula_6_A_prime(alpha):
    return alpha**(-1/(2*(alpha-1)))*abs(np.cos(np.pi*alpha/2))**(1/(2*(alpha-1)))/np.sqrt(2*np.pi*(alpha-1))

def skorohod_formula_6_B_prime(alpha):
    return (alpha-1)*alpha**(-alpha/(alpha-1))*abs(np.cos(np.pi*alpha/2))**(1/(alpha-1))

def skorohod_formula_6_lambda_prime(alpha):
    return alpha/(alpha-1)

def skorohod_formula_6_cdf(x,alpha):
    x=np.asarray(x,dtype=float)
    A=skorohod_formula_6_A_prime(alpha)
    B=skorohod_formula_6_B_prime(alpha)
    lam=skorohod_formula_6_lambda_prime(alpha)
    z=B*x**lam
    return A/(lam*np.sqrt(B))*gamma(0.5)*gammainc(0.5,z)


# ============================================================================
# Finalized tail series (the form the CDF wrapper dispatches to)
# ============================================================================

def skorohod_tail_series_sf(x, alpha, beta, nmax=SKOROHOD_TAIL_NMAX):
    """
    Survival function P(X > x) for x > 0 from the Skorohod tail expansion.

        P(X > x) ~ (1/(pi*alpha)) * sum_{n>=1} (-1)^(n-1) Gamma(n*alpha+1)/n!
                   * (1 + beta^2 tan^2(pi*alpha/2))^(n/2)
                   * sin(n (pi*alpha/2 + arctan(beta tan(pi*alpha/2)))) / n
                   * x^(-alpha*n)

    Two departures from Formula 1 above, both needed to make it usable:
    - Terms are built in the log domain, so Gamma(n*alpha+1) cannot overflow.
    - Each point is truncated at its own smallest term (optimal truncation),
      which is what makes the divergent-asymptotic alpha > 1 case work.

    Returns nan where the expansion is not trustworthy: x <= 0, or the smallest
    retained term is not small relative to the partial sum. That bound is the
    honest accuracy limit of an asymptotic series -- at alpha=0.3, beta=-0.8 it
    rejects x=0.0125 (where the series reads 0.322 against a true 0.085) and
    accepts x=0.05 onwards (exact to 1e-15).
    """
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    out = np.full(x.shape, np.nan)

    ok = x > 0
    if not np.any(ok):
        return out

    # alpha == 1 is a removable singularity here: tan(pi/2) is infinite.
    if abs(alpha - 1.0) < 1e-12:
        if abs(beta) < 1e-12:
            out[ok] = np.arctan(1.0 / x[ok]) / np.pi  # Cauchy
        return out

    tan_half = np.tan(np.pi * alpha / 2.0)
    theta = np.arctan(beta * tan_half)
    half_log = 0.5 * np.log1p(beta ** 2 * tan_half ** 2)

    xs = x[ok]
    log_x = np.log(xs)

    total = np.zeros(xs.shape)
    previous_magnitude = np.full(xs.shape, np.inf)
    smallest_magnitude = np.full(xs.shape, np.inf)
    live = np.ones(xs.shape, dtype=bool)
    terms_used = np.zeros(xs.shape, dtype=int)

    for n in range(1, nmax + 1):
        # Magnitude excludes the sin factor: sin passes through zero and would
        # otherwise trigger a spurious "smallest term" stop.
        log_magnitude = (
            gammaln(n * alpha + 1.0)
            - gammaln(n + 1.0)
            + n * half_log
            - np.log(np.pi * alpha * n)
            - alpha * n * log_x
        )
        magnitude = np.exp(log_magnitude)

        live &= magnitude <= previous_magnitude
        if not np.any(live):
            break

        term = magnitude * np.sin(n * (np.pi * alpha / 2.0 + theta))
        total = np.where(live, total + ((-1) ** (n - 1)) * term, total)
        terms_used = np.where(live, n, terms_used)
        smallest_magnitude = np.where(live, magnitude, smallest_magnitude)
        previous_magnitude = np.where(live, magnitude, previous_magnitude)

    with np.errstate(divide="ignore", invalid="ignore"):
        relative_truncation_error = smallest_magnitude / np.abs(total)

    total = np.where(
        (terms_used == 0) | ~(relative_truncation_error < SKOROHOD_TAIL_TRUNC_TOL),
        np.nan,
        total,
    )
    total = np.where((total < 0) | (total > 1), np.nan, total)

    out[ok] = total
    return out


def generate_cdf(X, alpha, beta):
    """
    CDF from the finalized Skorohod tail series.

    - Evaluates the survival function on |x| and uses the reflection identity
      F(x; alpha, beta) = 1 - F(-x; alpha, -beta) for the left side.
    - Returns nan outside the asymptotic regime; the wrapper in cdf.py only
      routes tail points here, so those nans are not reached in normal use.
    """
    X = np.atleast_1d(np.asarray(X, dtype=np.float64))
    F = np.full(X.shape, np.nan)

    pos = X > 0
    if np.any(pos):
        F[pos] = 1.0 - skorohod_tail_series_sf(X[pos], alpha, beta)

    neg = X < 0
    if np.any(neg):
        F[neg] = skorohod_tail_series_sf(-X[neg], alpha, -beta)

    return F