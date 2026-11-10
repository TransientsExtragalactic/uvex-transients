import matplotlib.pyplot as plt

from uvex_transients.models.supernovae import TypeIcSED
from uvex_transients.utils.plotting import plot_swift_uv_calibration

plot_swift_uv_calibration(TypeIcSED, "Ic", min_points=3, band="UVW2")
plt.show()