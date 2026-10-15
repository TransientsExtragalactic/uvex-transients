import matplotlib.pyplot as plt

from uvex_transients.models.supernovae import TypeIcBLSED
from uvex_transients.utils.plotting import plot_swift_uv_calibration

plot_swift_uv_calibration(TypeIcBLSED, "Ic-BL", min_points=3)
plt.show()