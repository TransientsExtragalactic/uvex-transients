import matplotlib.pyplot as plt

from uvex_transients.models.supernovae import ArnettMagnetarSpindownSED
from uvex_transients.utils.plotting import plot_swift_uv_calibration

plot_swift_uv_calibration(ArnettMagnetarSpindownSED, "SLSN-I", min_points=3, band="UVW1")
plt.show()