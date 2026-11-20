import matplotlib.pyplot as plt

from uvex_transients.models.supernovae import TypeIbSED
from uvex_transients.utils.plotting import plot_swift_uv_calibration

plot_swift_uv_calibration(TypeIbSED, "Ib", band="UVW1")
plt.show()