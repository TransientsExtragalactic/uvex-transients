"""
Cut: Sky Position Box
=========================

:meth:`~uvex_transients.simulation.core.SurveySimulator.filter_by_sky_position` keeps (or
drops) events whose position falls in a longitude/latitude box, in any coordinate frame
`~astropy.coordinates.SkyCoord.transform_to` accepts. Unlike
:meth:`~uvex_transients.simulation.core.SurveySimulator.filter_by_region` (arbitrary
polygon/circle regions), this is a simple coordinate-box test, e.g. for Galactic-plane
avoidance.
"""

from astropy import units as u
from m4opt.missions import uvex
from matplotlib import pyplot as plt

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
# Excluding the Galactic plane
# ----------------------------------
#
# ``frame="galactic", theta_min=-10, theta_max=10, mode="exclude"`` drops everything
# within 10 degrees of the Galactic plane, with `phi_min`/`phi_max` left at their
# full-sky defaults.

plane_avoided = simulator.filter_by_sky_position(
    catalog, uvex, frame="galactic", theta_min=-10, theta_max=10, mode="exclude"
)
print(f"{len(plane_avoided) * DOWNSAMPLE} events survive Galactic-plane avoidance.")

# %%
# Sky positions, before and after
# ----------------------------------------

galactic_all = catalog.coord.galactic
galactic_kept = plane_avoided.coord.galactic

fig, ax = resolve_fig_axes(fig_size=(8, 4), subplot_kw={"projection": "aitoff"})
ax.grid(True)
ax.scatter(
    galactic_all.l.wrap_at(180 * u.deg).radian, galactic_all.b.radian, s=2, alpha=0.2, color="#888888", label="All"
)
ax.scatter(
    galactic_kept.l.wrap_at(180 * u.deg).radian, galactic_kept.b.radian, s=4, color="#4C72B0", label="Plane avoided"
)
ax.legend(loc="lower right", markerscale=4)
ax.set_title("Effect of the sky-position cut (Galactic coordinates)")
fig.tight_layout()
plt.show()
