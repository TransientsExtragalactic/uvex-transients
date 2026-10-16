import numpy as np
import matplotlib.pyplot as plt

from uvex_transients.missions import uvex_fast as uvex
from uvex_transients.transients.kilonovae import Kilonova
from uvex_transients.utils.plotting import plot_detection_horizon

kilonova = Kilonova()
curve, grid = kilonova.get_detection_horizon(
    np.arange(22.0, 27.01, 0.5),
    uvex.detector.bandpasses,
    z_min=kilonova.redshift_limit / 4,
    z_max=4 * kilonova.redshift_limit,
    n_samples=500,
    rng=0,
    progress=False,
)
print(curve)

plot_detection_horizon(kilonova, curve, grid, mag_limit=24.5)
plt.show()