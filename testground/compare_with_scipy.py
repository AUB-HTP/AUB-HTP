import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levy_stable
from aub_htp import generate_alpha_stable_pdf

alpha = 1
beta = 1
gamma = 1
delta = 0

x_vals = np.linspace(-10, 10, 1000)
y = generate_alpha_stable_pdf(x_vals, alpha, beta, gamma, delta)
y_scipy = levy_stable.pdf(x_vals,
                          alpha = alpha,
                          beta = beta,
                          loc = delta,
                          scale = gamma)
# Plotting

plt.plot(x_vals, y, label="Custom Stable PDF", linewidth=2)
plt.plot(x_vals, y_scipy, label="Scipy Stable PDF", linewidth=2)
plt.title(f"Stable PDF (alpha={alpha}, beta={beta})")
plt.xlabel("x")
plt.ylabel("PDF")
plt.legend()
plt.grid(True)

plt.show()