import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.time import Time

from uvex_transients.models.core import ComposedSpectralModel
from uvex_transients.models.lightcurves.generic import GaussianRiseBrokenPowerLawLightcurve
from uvex_transients.models.spectra.thermal import BlackbodySpectrum
from uvex_transients.transients.base import ExtragalacticTransient


class ToyNovaSED(ComposedSpectralModel):
    _LIGHTCURVE_CLASS = GaussianRiseBrokenPowerLawLightcurve
    _SPECTRUM_CLASS = BlackbodySpectrum


_TOY_NOVA_RATE = 1e4 / (u.Gpc**3 * u.yr)


class ToyNova(ExtragalacticTransient):
    DEFAULT_MODEL = ToyNovaSED
    DEFAULT_DURATION = 20 * u.day
    DEFAULT_Z_LIM = 0.05

    def event_rate(self, z):
        z = np.asarray(z)
        rate = np.full_like(z, _TOY_NOVA_RATE.to_value(u.Mpc**-3 * u.yr**-1), dtype=np.float64)
        return rate if z.ndim > 0 else rate.item()

nova = ToyNova()

events = nova.sample_events_on_healpix_grid(
    nside=32, t_start=Time("2025-01-01"), duration=365 * u.day, seed=0,
)

fig = plt.figure(figsize=(7, 4))
ax = fig.add_subplot(111, projection="aitoff")
ax.grid(True)
ra = events["coord"].ra.wrap_at(180 * u.deg).radian
ax.scatter(ra, events["coord"].dec.radian, s=4, c=events["redshift"], cmap="plasma")
ax.set_title(f"{len(events)} sampled ToyNova events, whole sky, one year")