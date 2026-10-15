import matplotlib.pyplot as plt

from uvex_transients.models.supernovae import TypeIIbSED
from uvex_transients.utils.plotting import plot_swift_uv_calibration

plot_swift_uv_calibration(TypeIIbSED, "IIb")
plt.show()