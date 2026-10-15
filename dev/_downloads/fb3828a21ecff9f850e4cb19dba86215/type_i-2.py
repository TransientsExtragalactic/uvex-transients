import matplotlib.pyplot as plt

from uvex_transients.models.supernovae import TypeIaSED
from uvex_transients.utils.plotting import plot_swift_uv_calibration

plot_swift_uv_calibration(TypeIaSED, "Ia")
plt.show()