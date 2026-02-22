import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levy_stable
import aub_htp as ht

# =========================
# PARAMETERS
# =========================
alpha = 0.9          # stability parameter (0 < alpha <= 2)
beta = 0.5           # skewness parameter (-1 <= beta <= 1)
loc = 10
scale = 1000
n_samples = 500000

levy_stable.parameterization = 'S0'
ht.alpha_stable.with_parametrization("S0")

# =========================
# SAMPLE FROM SCIPY
# =========================
scipy_samples_all = levy_stable.rvs(alpha, beta, loc=loc, scale=scale, size=n_samples)

# =========================
# CUSTOM ALPHA-STABLE ARRAY
# =========================
X_all = ht.alpha_stable.rvs(alpha, beta, loc = loc, scale = scale, size=n_samples)

# =========================
# TRUNCATION RANGE
# =========================
delta = np.array([loc])
gamma = scale
xmin = -10000 + delta[0]
xmax = 10000 + delta[0]

# Truncate arrays
scipy_samples = scipy_samples_all[(scipy_samples_all >= xmin) & (scipy_samples_all <= xmax)]
X_trunc = X_all[(X_all >= xmin) & (X_all <= xmax)]

# =========================
# FIGURE 1: SCIPY HISTOGRAM + PDF
# =========================
plt.figure(figsize=(10,5))
counts_scipy, bins_scipy, _ = plt.hist(
    scipy_samples,
    bins=2000,
    density=True,
    alpha=0.5,
    color="#00e5ff",
    edgecolor="black",
    label="Truncated SciPy Alpha-Stable"
)
bin_centers_scipy = (bins_scipy[:-1] + bins_scipy[1:]) / 2

# Evaluate PDF at bin centers
pdf_scipy_at_bins = levy_stable.pdf(bin_centers_scipy, alpha, beta, loc=delta, scale=gamma)

# Renormalize PDF for truncation
F_xmax = levy_stable.cdf(xmax, alpha, beta, loc=delta, scale=gamma)
F_xmin = levy_stable.cdf(xmin, alpha, beta, loc=delta, scale=gamma)
pdf_scipy_trunc = pdf_scipy_at_bins / (F_xmax - F_xmin)

# Overlay PDF
plt.plot(bin_centers_scipy, pdf_scipy_trunc, 'black', lw=2, label="Truncated SciPy PDF")
plt.title(f"Truncated SciPy Histogram with PDF Overlay (param={levy_stable.parameterization})")
plt.xlabel("Value")
plt.ylabel("Density")
plt.xlim(xmin, xmax)
plt.legend()
plt.grid(True)
plt.show(block=False)

# =========================
# FIGURE 2: CUSTOM HISTOGRAM + PDF
# =========================
plt.figure(figsize=(10,5))
counts_custom, bins_custom, _ = plt.hist(
    X_trunc,
    bins=2000,
    density=True,
    alpha=0.5,
    color="#00e5ff",
    edgecolor="black",
    label="Custom Truncated Array"
)
bin_centers_custom = (bins_custom[:-1] + bins_custom[1:]) / 2

# Evaluate PDF at bin centers
pdf_custom_at_bins = levy_stable.pdf(bin_centers_custom, alpha, beta, loc=delta, scale=gamma)

# Renormalize PDF for truncation
pdf_custom_trunc = pdf_custom_at_bins / (F_xmax - F_xmin)

# Overlay PDF
plt.plot(bin_centers_custom, pdf_custom_trunc, 'black', lw=2, label="Truncated SciPy PDF")
plt.title(f"Custom Truncated Histogram with PDF Overlay (param={levy_stable.parameterization})")
plt.xlabel("Value")
plt.ylabel("Density")
plt.xlim(xmin, xmax)
plt.legend()
plt.grid(True)
plt.savefig("alpha_stable.png")