import matplotlib.pyplot as plt

from uvex_transients.transients.supernovae import TypeIbSNe
from uvex_transients.utils.filter_utils import get_bandpass
from uvex_transients.utils.plotting import plot_luminosity_function

v_band = get_bandpass("Generic/Johnson.V")
lf = TypeIbSNe().luminosity_function(v_band, 10000, quantity="magnitude", bins=30, rng=0)
plot_luminosity_function(lf, band="V")
plt.show()