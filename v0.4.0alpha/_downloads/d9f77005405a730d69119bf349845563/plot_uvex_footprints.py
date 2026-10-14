"""
UVEX Survey Footprints
======================

UVEX's footprints come from the DS9 region files in the
`uvex-scheduler <https://github.com/m4opt/uvex-scheduler/tree/main/survey-footprints>`_
repo. The first use of each downloads its file and caches the resulting MOC, so later runs
read it from disk.
"""

# %%
# Each footprint is looked up by name in the
# :data:`~uvex_transients.surveys.footprints.base.default_registry`, which every
# footprint registers itself in on import.

import matplotlib.pyplot as plt

from uvex_transients.surveys.footprints import default_registry
from uvex_transients.utils.plotting import plot_footprints

print(default_registry.names("uvex:"))

# %%
# The low-Milky-Way/low-zodi wide survey, its deep fields, and the Magellanic Clouds
# survey. The deep fields are small circles (the inscribed UVEX field of view) and larger
# patches.

footprints = {
    "LMLZ wide": default_registry["uvex:lmlz:wide"],
    "LMLZ deep": default_registry["uvex:lmlz:deep"],
    "Magellanic Clouds": default_registry["uvex:mc"],
}
plot_footprints(footprints, title="UVEX survey footprints")
plt.show()
