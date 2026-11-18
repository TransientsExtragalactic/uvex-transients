import matplotlib.pyplot as plt

from uvex_transients.models.supernovae import TypeIIPSED
from uvex_transients.utils.plotting import plot_swift_uv_calibration

plot_swift_uv_calibration(TypeIIPSED, "IIP", band="UVW2")
plt.show()