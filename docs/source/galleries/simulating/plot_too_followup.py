"""
Target-of-Opportunity Follow-up of a TDE
===========================================

:ref:`sphx_glr_auto_examples_simulating_plot_tde_end_to_end.py` starts from a whole
population, Monte Carlo sampled and screened against a real
:class:`~uvex_transients.surveys.base.SurveySchedule`. A target-of-opportunity (ToO)
follow-up is the opposite problem: one *specific* TDE, at a position and redshift
already known (from an alert, say), that gets pointed at directly on whatever cadence
is chosen -- no schedule, no windowed sampling, no
:class:`~uvex_transients.simulation.event.Event`.

:meth:`~uvex_transients.models.core.base.SpectralModel.simulate_photometry` is built
for exactly this: given a sky position and whatever time grid and exposure time the
caller wants evaluated, it runs the same noise model a real survey simulation uses
(a batched :class:`~synphot.SourceSpectrum` plus
:meth:`~m4opt.synphot.Detector.get_snr`), with no schedule in the loop. This example
follows one TDE weekly across its whole ~200-day duration and builds its UV light
curves.
"""

# %%
# Choosing the target
# ----------------------
#
# A target of opportunity is a specific object, not a population draw.
# :class:`~uvex_transients.transients.TDEs.TidalDisruptionEvent` still supplies the
# right SED model (:class:`~uvex_transients.models.tdes.van_velzen.VanVelzenTDESED`)
# and a conservative duration window; only its *parameters* are fixed here, to one
# concrete realization (via a seed, for reproducibility), rather than left to be
# sampled for a whole population. Milky Way foreground dust is resolved the same way
# a real survey simulation resolves it -- :func:`~uvex_transients.dust.resolve_ebv`
# against the real sky map -- since it depends only on sky position, not on when the
# TDE happens to be observed.

from functools import partial

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from m4opt.missions import uvex
from m4opt.synphot.background import GalacticBackground
from matplotlib import pyplot as plt

from uvex_transients.dust import dust_map, log_attenuation, resolve_ebv
from uvex_transients.transients.TDEs import TidalDisruptionEvent

tde = TidalDisruptionEvent()

coord = SkyCoord(ra=195.3 * u.deg, dec=27.8 * u.deg)
redshift = 0.03
luminosity_distance = tde.cosmology.luminosity_distance(redshift)
ebv = float(resolve_ebv(dust_map(), coord))

params = {name: value[0] for name, value in tde.sed.sample_parameters(1, rng=12345).items()}
print(f"E(B-V) at target: {ebv:.3f}")
print("SED parameters:", {name: f"{value:.3g}" for name, value in params.items()})

# %%
# A weekly cadence
# -------------------
#
# No schedule to query -- just pick the times to observe. :attr:`tde.duration_limit
# <uvex_transients.transients.base.TransientBase.duration_limit>` is a conservative
# upper bound on how long this class of event stays relevant; weekly visits across
# that whole window is a plausible real follow-up cadence for something this slow.

CADENCE = 7 * u.day
EXPTIME = 900 * u.s

t = np.arange(0, tde.duration_limit.to_value(u.day), CADENCE.to_value(u.day)) * u.day
print(f"{len(t)} visits, one every {CADENCE}.")

# %%
# Simulating photometry, with no zodiacal light
# --------------------------------------------------
#
# A real ToO trigger time isn't known in advance, so there's no meaningful
# ``obstime`` to feed a season-dependent background term like zodiacal light. Passing
# ``background=GalacticBackground()`` explicitly simulates against dust (already
# folded into the source flux itself, via ``log_attenuation``) plus the Milky Way's
# diffuse UV glow only, and leaves ``observer_location``/``obstime`` at their
# placeholder defaults -- both irrelevant to
# :class:`~m4opt.synphot.background.GalacticBackground`, so there is nothing else to
# supply. See :meth:`~uvex_transients.models.core.base.SpectralModel.simulate_photometry`'s
# own docstring for exactly when that placeholder default is (and isn't) safe.

phot = tde.sed.simulate_photometry(
    t,
    EXPTIME,
    uvex.detector,
    coord,
    background=GalacticBackground(),
    redshift=redshift,
    luminosity_distance=luminosity_distance,
    log_attenuation=partial(log_attenuation, Ebv=ebv),
    rng=0,
    **params,
)
print(phot["t", "band", "snr", "ab_mag"][:6])

# %%
# The light curves
# -------------------
#
# The noiseless theory curve (:meth:`~uvex_transients.models.core.base.SpectralModel.mag`,
# evaluated at each band's pivot wavelength) alongside the simulated weekly visits:
# detections above SNR=5 as points with error bars, fainter visits as
# downward-pointing upper limits -- the same plotting convention used at the end of
# :ref:`sphx_glr_auto_examples_simulating_plot_tde_end_to_end.py`.

SNR_THRESHOLD = 5.0
t_theory = np.linspace(0, tde.duration_limit.to_value(u.day), 300) * u.day

fig, ax = plt.subplots(figsize=(7, 4))
for band, color in {"FUV": "#4C72B0", "NUV": "#DD8452"}.items():
    nu = uvex.detector.bandpasses[band].pivot().to(u.Hz, equivalencies=u.spectral())
    theory_mag = tde.sed.mag(
        nu,
        t_theory,
        redshift=redshift,
        luminosity_distance=luminosity_distance,
        log_attenuation=log_attenuation(nu, ebv),
        **params,
    )
    ax.plot(t_theory.value, theory_mag.value, color=color, lw=1.5, alpha=0.6)

    in_band = np.isfinite(phot["ab_mag"]) & (phot["band"] == band)
    detected = in_band & (phot["snr"] > SNR_THRESHOLD)
    upper_limits = in_band & (phot["snr"] <= SNR_THRESHOLD)

    if np.any(detected):
        ax.errorbar(
            phot["t"][detected].to_value(u.day),
            phot["ab_mag"][detected],
            yerr=5 * phot["mag_err"][detected],
            marker="s",
            mfc=color,
            mec="k",
            ecolor=color,
            linestyle="none",
            label=band,
        )
    if np.any(upper_limits):
        ax.errorbar(
            phot["t"][upper_limits].to_value(u.day),
            phot["ab_mag"][upper_limits],
            yerr=[
                phot["mag_upper"][upper_limits] - phot["ab_mag"][upper_limits],
                np.abs(phot["mag_lower"][upper_limits] - phot["ab_mag"][upper_limits]),
            ],
            marker="v",
            mfc="w",
            mec=color,
            ecolor=color,
            linestyle="none",
        )

ax.invert_yaxis()
ax.set_xlabel("Days since explosion")
ax.set_ylabel("AB magnitude")
ax.set_title(f"Weekly ToO follow-up (z={redshift}, {len(t)} visits)")
ax.legend()
fig.tight_layout()

# %%
# Bonus: what zodiacal light would have cost
# -----------------------------------------------
#
# Suppose the trigger date *had* been known -- say, this TDE actually went off on a
# specific night. Comparing that against the Galactic-only run above shows what
# assuming ``background=GalacticBackground()`` glossed over: passing
# ``background=None`` (the default) uses :data:`~m4opt.missions.uvex`'s own detector
# background, Galactic *and* zodiacal light together, and now real
# ``observer_location``/``obstime`` values (from
# :meth:`~m4opt.missions.uvex.observer_location`) are needed too, since zodiacal
# light actually depends on both.

hypothetical_trigger = Time("2031-03-01T00:00:00", scale="utc")
obstime = hypothetical_trigger + t
observer_location = uvex.observer_location(obstime)

phot_with_zodi = tde.sed.simulate_photometry(
    t,
    EXPTIME,
    uvex.detector,
    coord,
    observer_location=observer_location,
    obstime=obstime,
    redshift=redshift,
    luminosity_distance=luminosity_distance,
    log_attenuation=partial(log_attenuation, Ebv=ebv),
    rng=0,
    **params,
)

for band in uvex.detector.bandpasses:
    galactic_only = np.nanmedian(phot["snr"][phot["band"] == band])
    with_zodi = np.nanmedian(phot_with_zodi["snr"][phot_with_zodi["band"] == band])
    print(f"{band}: median SNR {galactic_only:.1f} (Galactic only) vs {with_zodi:.1f} (+ zodiacal)")
