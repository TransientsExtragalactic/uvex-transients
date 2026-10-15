import numpy as np
import matplotlib.pyplot as plt

from uvex_transients.missions import uvex_fast as uvex
from uvex_transients.transients.supernovae import TypeIIPExcessSNe
from uvex_transients.utils.plotting import plot_detection_horizon

transient = TypeIIPExcessSNe()
curve, grid = transient.get_detection_horizon(
    np.arange(22.0, 27.01, 0.5),
    uvex.detector.bandpasses,
    z_min=transient.redshift_limit / 4,
    z_max=4 * transient.redshift_limit,
    n_z=45,
    rng=0,
    progress=False,
)
plot_detection_horizon(transient, curve, grid, mag_limit=24.5)
plt.show()