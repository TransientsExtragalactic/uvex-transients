"""
Cut: Sky Region
===================

:meth:`~uvex_transients.simulation.core.SurveySimulator.filter_by_region` keeps only
events inside an arbitrary sky region: a `~regions.SkyRegion`/`~regions.Regions`
instance, or a path to a region file (e.g. a DS9 ``.reg``). It uses the same
``regions``/:func:`~m4opt.fov.contains` convention
`~uvex_transients.surveys.base.SurveySchedule` uses for its own instrument FOV, so no WCS
is needed.
"""

from astropy import units as u
from astropy.coordinates import SkyCoord
from m4opt.missions import uvex
from matplotlib import pyplot as plt
from regions import CircleSkyRegion

from uvex_transients.simulation.core import SurveySimulator
from uvex_transients.surveys import get_schedule
from uvex_transients.transients.supernovae import MagnetarSLSNe
from uvex_transients.utils.plotting import resolve_fig_axes, set_plot_style

set_plot_style()

schedule = get_schedule()
slsn = MagnetarSLSNe()
simulator = SurveySimulator(schedule, transients={"slsn": slsn}, simulation_seed=42)

TIME_BINS = 10
NSIDE = 32
DOWNSAMPLE = 50

catalog = simulator.generate_events(time_bins=TIME_BINS, nside=NSIDE, downsample=DOWNSAMPLE)
print(f"{len(catalog) * DOWNSAMPLE} events sampled.")

# %%
# A circular region around the median sampled position
# ------------------------------------------------------------
#
# Any `~regions.SkyRegion` works; a circle is the simplest.

center = SkyCoord(ra=catalog.coord.ra.deg.mean() * u.deg, dec=catalog.coord.dec.deg.mean() * u.deg)
region = CircleSkyRegion(center=center, radius=20 * u.deg)

in_region = simulator.filter_by_region(catalog, uvex, region=region)
print(f"{len(in_region) * DOWNSAMPLE} events fall within {region.radius} of {center}.")

# %%
# Sky positions, in and out of the region
# ----------------------------------------------

fig, ax = resolve_fig_axes(fig_size=(8, 4), subplot_kw={"projection": "aitoff"})
ax.grid(True)
ax.scatter(
    catalog.coord.ra.wrap_at(180 * u.deg).radian, catalog.coord.dec.radian, s=2, alpha=0.2, color="#888888", label="All"
)
ax.scatter(
    in_region.coord.ra.wrap_at(180 * u.deg).radian,
    in_region.coord.dec.radian,
    s=4,
    color="#4C72B0",
    label="In region",
)
ax.legend(loc="lower right", markerscale=4)
ax.set_title("Effect of the region cut")
fig.tight_layout()
plt.show()
