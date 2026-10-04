"""
LS4 SOLE Survey Footprint
=========================

LS4 (`Bellm et al. 2025 <https://arxiv.org/abs/2503.14579>`_) has no published footprint
file. Its SOLE survey (Stellar Oscillations, Lensing, and Eruptions) is defined in the
paper as two boxes in Galactic coordinates, a plane strip and a bulge region, and the
registered ``ls4:sole`` footprint is their union.
"""

# %%
# The two boxes live in the config, and `galactic_box_MOC` builds each one.

import matplotlib.pyplot as plt

from uvex_transients.surveys.footprints import default_registry
from uvex_transients.surveys.footprints.utils import galactic_box_MOC
from uvex_transients.utils import config
from uvex_transients.utils.plotting import plot_footprints

plane, bulge = (galactic_box_MOC(max_order=10, **box) for box in config["observatories.ls4.sole.boxes"])

plot_footprints({"Galactic plane": plane, "Galactic bulge": bulge}, title="LS4 SOLE regions")
plt.show()

# %%
# Their union, as registered, is slightly smaller than the sum of the two areas, since the
# bulge box overlaps the plane strip.

plot_footprints({"ls4:sole": default_registry["ls4:sole"]}, title="LS4 SOLE footprint")
plt.show()
