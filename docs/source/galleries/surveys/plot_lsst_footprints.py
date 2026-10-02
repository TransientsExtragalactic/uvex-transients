"""
LSST Survey Footprints
======================

The LSST footprints are built from the Rubin scheduler's own labeled sky-area map and
deep drilling field list, using the optional
`rubin_scheduler <https://rubin-scheduler.lsst.io>`_ package
(``pip install "uvex-transients[rubin]"``). Generating a MOC downloads the scheduler data
once through astropy's cache. Each finished MOC is then cached, so later runs need neither
the package nor the download.
"""

# %%
# Each region of the scheduler's sky-area map is its own footprint, named ``lsst:*``.

import matplotlib.pyplot as plt

from uvex_transients.surveys.footprints import default_registry
from uvex_transients.utils.plotting import plot_footprints

print(default_registry.names("lsst:"))

# %%
# The wide-fast-deep area covers most of the survey, and the deep drilling fields are drawn
# last, on top. Colors are assigned in order, so the footprints are ordered to keep regions
# that touch each other well separated.

footprints = {
    "Wide-fast-deep": default_registry["lsst:wfd"],
    "Galactic plane and bulge": default_registry["lsst:galplane"],
    "North ecliptic spur": default_registry["lsst:nes"],
    "Magellanic Clouds": default_registry["lsst:mc"],
    "Virgo cluster": default_registry["lsst:virgo"],
    "South celestial pole": default_registry["lsst:scp"],
    "Deep drilling fields": default_registry["lsst:ddf"],
}
plot_footprints(footprints, title="LSST survey footprints")
plt.show()
