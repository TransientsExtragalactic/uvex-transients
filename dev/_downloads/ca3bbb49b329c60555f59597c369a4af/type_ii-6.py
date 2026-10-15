import matplotlib.pyplot as plt

from uvex_transients.models.supernovae import TypeIIPExcessSED
from uvex_transients.utils.plotting import plot_swift_uv_calibration

plot_swift_uv_calibration(TypeIIPExcessSED, "IIP+Excess")
plt.show()