import matplotlib.pyplot as plt

from uvex_transients.transients.supernovae import TypeIbSNe
from uvex_transients.utils.filter_utils import get_bandpass
from uvex_transients.utils.plotting import plot_archived_luminosity_function, plot_luminosity_function

v_band = get_bandpass("Generic/Johnson.V")
lf = TypeIbSNe().luminosity_function(v_band, 10000, quantity="magnitude", bins=30, rng=0)
fig, ax = plot_luminosity_function(lf, band="V", label="Model")
plot_archived_luminosity_function(lf, "supernovae/Ib", "ASAS-SN", ax=ax)
plt.show()