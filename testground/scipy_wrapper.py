import numpy as np
import aub_htp as ht
import matplotlib.pyplot as plt
from scipy.stats import norm
dist = ht.alpha_stable()

alpha = 1
beta = 0.7
gamma = 2
delta = 1.5

x_vals = np.linspace(-10, 10, 30)
y = dist.pdf(x_vals, alpha, beta, scale = gamma, loc = delta)

print(x_vals)
print(y)
# Plotting

plt.plot(x_vals, y, label="Custom Stable PDF", linewidth=2)
plt.title(f"Stable PDF (alpha={alpha}, beta={beta})")
plt.xlabel("x")
plt.ylabel("PDF")
plt.legend()
plt.grid(True)

plt.show()