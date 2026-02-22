import aub_htp as ht
from scipy.stats import levy_stable
from scipy.stats import norm

x = [1,2,3,4]

y = ht.alpha_stable.pdf(0, alpha = [0.2,2], beta = 1, loc = 1, scale = 3)

z = norm.pdf(0, loc = [1,2], scale = 3)
print(y)
print(z)

