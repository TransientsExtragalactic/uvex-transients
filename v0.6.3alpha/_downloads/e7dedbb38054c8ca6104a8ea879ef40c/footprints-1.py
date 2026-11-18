import matplotlib.pyplot as plt

from uvex_transients.surveys.footprints import default_registry
from uvex_transients.utils.plotting import plot_footprints

ztf = default_registry["ztf:main"]

# Is (RA, Dec) = (10, 45) deg in ZTF's footprint? What about (10, -60)?
print(ztf.contains([10, 10], [45, -60]))

plot_footprints({"ZTF": ztf}, title="ZTF survey footprint")
plt.show()