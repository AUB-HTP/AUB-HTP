import numpy as np
import matplotlib.pyplot as plt
from aub_htp import generate_alpha_stable_pdf

alpha = 1.4
beta = 0.9
gamma = 0.5
delta = 1

x_vals = np.linspace(-10, 10, 4000)
y = generate_alpha_stable_pdf(x_vals, alpha, beta, gamma, delta)

# Plotting

plt.plot(x_vals, y, label="Custom Stable PDF", linewidth=2)
plt.title(f"Stable PDF (alpha={alpha}, beta={beta})")
plt.xlabel("x")
plt.ylabel("PDF")
plt.legend()
plt.grid(True)

plt.show()