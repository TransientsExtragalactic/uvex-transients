import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from uvex_transients.missions import uvex_fast as uvex
from uvex_transients.transients.supernovae import TypeIIbSNe
from uvex_transients.utils.plotting import plot_detection_horizon

transient = TypeIIbSNe()
curve, grid = transient.get_detection_horizon(
    np.arange(22.0, 27.01, 0.5),
    uvex.detector.bandpasses,
    t_min=0.1 * u.day,
    z_min=transient.redshift_limit / 4,
    z_max=4 * transient.redshift_limit,
    n_z=45,
    rng=0,
    progress=False,
)
plot_detection_horizon(transient, curve, grid, mag_limit=24.5)
plt.show()