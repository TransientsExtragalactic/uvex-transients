import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import vstack
from m4opt.missions import rubin, uvex
from m4opt.synphot.background import GalacticBackground, SkyBackground
import matplotlib.pyplot as plt

from uvex_transients.dust import dust_map, log_attenuation, resolve_ebv
from uvex_transients.models.supernovae import MoragShockCoolingSED

sed = MoragShockCoolingSED()
coord = SkyCoord(ra=195.3 * u.deg, dec=27.8 * u.deg)
redshift = 0.02
distance = 90.0 * u.Mpc
ebv = float(resolve_ebv(dust_map(), coord))

param_draws = [
    {name: value[0] for name, value in sed.sample_parameters(1, rng=42).items()},
    {name: value[0] for name, value in sed.sample_parameters(1, rng=7).items()},
]

RUBIN_BANDS = ["g", "r", "i"]
CADENCE = 0.5 * u.day
RUBIN_EXPTIME = 30 * u.s
UVEX_EXPTIME = 900 * u.s
DURATION = 12 * u.day
SNR_THRESHOLD = 5.0

# Rubin/LSST's own photometric-calibration floor (Table 14 of the LSST Science Requirements
# Document, LPM-17), added in quadrature to the shot-noise-only default; UVEX is left as pure
# shot noise, with no comparably-established floor here.
RUBIN_SIGMA_SYS = {"g": 0.005, "r": 0.005, "i": 0.005}

band_detectors = {b: rubin.detector for b in RUBIN_BANDS} | {"FUV": uvex.detector, "NUV": uvex.detector}
band_colors = {"g": "#008060", "r": "#FF4000", "i": "#850000", "FUV": "#4C72B0", "NUV": "#DD8452"}

fig, axes = plt.subplots(1, len(param_draws), figsize=(11, 5), sharey=True)

for ax, params in zip(axes, param_draws):
    t_rubin = np.arrange(0.1, DURATION.to_value(u.day), CADENCE.to_value(u.day)) * u.day
    phot_rubin = sed.simulate_photometry(
        t_rubin, RUBIN_EXPTIME, rubin.detector, coord,
        bands=RUBIN_BANDS, background=SkyBackground.medium(),
        redshift=redshift, luminosity_distance=distance, ebv=ebv,
        sys_err=RUBIN_SIGMA_SYS, rng=0, **params,
    )

    t_uvex = np.arrange(0.1, DURATION.to_value(u.day), CADENCE.to_value(u.day)) * u.day
    phot_uvex = sed.simulate_photometry(
        t_uvex, UVEX_EXPTIME, uvex.detector, coord,
        background=GalacticBackground(),
        redshift=redshift, luminosity_distance=distance, ebv=ebv, rng=0, **params,
    )

    phot = vstack([phot_rubin, phot_uvex])

    t_theory = np.linspace(0.02, DURATION.to_value(u.day), 300) * u.day
    for band, color in band_colors.items():
        detector = band_detectors[band]
        nu = detector.bandpasses[band].pivot().to(u.Hz, equivalencies=u.spectral())
        theory_mag = sed.mag(
            nu, t_theory, redshift=redshift, luminosity_distance=distance,
            log_attenuation=log_attenuation(nu, ebv), **params,
        )
        ax.plot(t_theory.value, theory_mag.value, color=color, lw=1.2, alpha=0.6)

        in_band = np.isfinite(phot["ab_mag"]) & (phot["band"] == band)
        detected = in_band & (phot["snr"] > SNR_THRESHOLD)
        upper_limits = in_band & (phot["snr"] <= SNR_THRESHOLD)

        if np.any(detected):
            ax.errorbar(
                phot["t"][detected].to_value(u.day), phot["ab_mag"][detected],
                yerr=phot["mag_err"][detected], marker="s", mfc=color, mec="k",
                ecolor=color, linestyle="none", label=band, ms=4,
            )
        if np.any(upper_limits):
            ax.errorbar(
                phot["t"][upper_limits].to_value(u.day), phot["ab_mag"][upper_limits],
                yerr=[phot["mag_upper"][upper_limits] - phot["ab_mag"][upper_limits],
                      np.abs(phot["mag_lower"][upper_limits] - phot["ab_mag"][upper_limits])],
                marker="v", mfc="w", mec=color, ecolor=color, linestyle="none", ms=4,
            )

    ax.invert_yaxis()
    ax.set_xlabel("Days since explosion")
    ax.set_ylim([26, 17])

axes[0].set_ylabel("AB magnitude")
axes[0].legend(ncol=3, fontsize=8)
fig.suptitle(f"Shock-cooling IIb: simulated Rubin+UVEX light curves (z={redshift})")
fig.tight_layout()