import numpy as np
import matplotlib.pyplot as plt
from aub_htp import generate_alpha_stable_pdf

alpha = 0.999999999
beta = 1
gamma = 1
delta = 0

x_vals = np.linspace(-10, 10, 10000)
y = generate_alpha_stable_pdf(x_vals, alpha, beta, gamma, delta)

# Plotting

plt.plot(x_vals, y, label="Custom Stable PDF", linewidth=2)
plt.title(f"Stable PDF (alpha={alpha}, beta={beta})")
plt.xlabel("x")
plt.ylabel("PDF")
plt.legend()
plt.grid(True)

plt.show()