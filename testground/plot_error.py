from aub_htp.random.alpha_stable_sampler import estimate_number_of_convergence_terms
import numpy as np

alpha0 = np.linspace(0.01, 0.99, 1000)
alpha1 = np.linspace(1, 1.99, 1000)
alpha = np.concatenate([alpha0, alpha1])
y = [min(np.log10(estimate_number_of_convergence_terms(0.01,alpha_,1)), 4) for alpha_ in alpha]

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(alpha, y)
plt.savefig("alpha_stable.png")