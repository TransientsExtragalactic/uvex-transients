"""
ZTF Survey Footprint
====================

ZTF has no published region file. Its coverage is the sky visible from Palomar, which a
declination cut (``config["observatories.ztf.min_dec"]``) describes well.
"""

# %%
import matplotlib.pyplot as plt

from uvex_transients.surveys.footprints import default_registry
from uvex_transients.utils.plotting import plot_footprints

ztf = default_registry["ztf:main"]
print(ztf.description)

plot_footprints({"ZTF": ztf}, title="ZTF survey footprint")
plt.show()
