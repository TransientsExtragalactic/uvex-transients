import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

C, delta, M, n_repeats = 0.95, 0.01, 1000, 5000
rank = int(np.count_nonzero(binom.cdf(np.arange(M + 1), M, delta) <= 1 - C))

# Draws that are uniform on (0, 1) have a known tail: the fraction of the population beyond a
# value u is exactly 1 - u.
rng = np.random.default_rng(0)
draws = np.sort(rng.random((n_repeats, M)), axis=1)
true_tail_fraction = 1 - draws[:, M - rank]

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(true_tail_fraction * 100, bins=60, color="0.6")
ax.axvline(delta * 100, color="C3", label="tolerance, 1%")
covered = np.mean(true_tail_fraction <= delta)
ax.set_xlabel("True percentage of the population beyond the chosen limit")
ax.set_ylabel("Repeats")
ax.set_title(f"{covered:.1%} of repeats are at or below the tolerance")
ax.legend(frameon=False)
fig.tight_layout()
plt.show()