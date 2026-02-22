import numpy as np
import aub_htp as ht
from aub_htp import generate_alpha_stable_pdf
from scipy.stats import levy_stable
import matplotlib.pyplot as plt

alpha = 1
beta = 0.9
gamma = 1.5
delta = 2

# + (2 / np.pi) * beta * gamma * np.log(gamma)
x_vals = np.linspace(-10, 10, 1000)
y_raw = generate_alpha_stable_pdf(x_vals,
                                  alpha, beta, gamma, delta)
y_wrap = ht.alpha_stable.with_parametrization("S1").pdf(x_vals,
                  alpha = alpha,
                  beta = beta,
                  scale = gamma,
                  loc = delta)
y_scipy = levy_stable.pdf(x_vals,
                          alpha, beta, loc = delta, scale = gamma)

# Plotting
print(ht.alpha_stable.parameterization)
plt.plot(x_vals, y_wrap, label="Scipy Wrapper", linewidth=2)
plt.plot(x_vals, y_raw, label="Paper", linewidth=2)
plt.plot(x_vals, y_scipy, label="Scipy LevyStable", linewidth=2)
plt.title(f"Stable PDF (alpha={alpha}, beta={beta})")
plt.xlabel("x")
plt.ylabel("PDF")
plt.legend()
plt.grid(True)

plt.show()