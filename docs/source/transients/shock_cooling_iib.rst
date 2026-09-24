.. _transients_shock_cooling_iib:

Shock-Cooling Emission from Type IIb SNe
==========================================

Type IIb supernovae are core-collapse explosions of massive stars that have been stripped of most,
but not all, of their hydrogen envelope. Many show a double-peaked light curve: an early, hours-
to-days-long flash powered by the shock heating and subsequent cooling of the extended envelope,
followed -- after a dip -- by a broader, weeks-long peak powered by radioactive :math:`^{56}\mathrm{Ni}`
decay, the same mechanism that powers most other core-collapse SN light curves. This population
models only the first of those two components: the early shock-cooling emission, using the
semi-analytic diffusion-envelope model of :footcite:t:`2024MNRAS.528.7137M`.

This population is implemented by
:class:`~uvex_transients.transients.supernovae.ShockCoolingIIb`, pairing
:class:`~uvex_transients.models.supernovae.IIb.MoragShockCoolingSED` with the rate/duration
metadata described below. It is deliberately **not** a full Type IIb light curve model -- see
Quick Facts below for why its duration window is so much shorter than the other supernova
populations in this package.

Quick Facts
------------

.. list-table::
   :header-rows: 1
   :widths: 15 25 15 45

   * - Quantity
     - Value
     - Source
     - Notes
   * - Rate
     - :math:`R_\mathrm{CC}(z) = k\,\psi_\mathrm{UV}(z)`; Type IIb 10.3% of :math:`R_\mathrm{CC}(z)`
     - :footcite:t:`strolger2015`, :footcite:t:`madau2014`, :footcite:t:`shivvers2017`
     - Tracks the cosmic star-formation history. :footcite:t:`shivvers2017` find IIb is
       :math:`34.0\pm11.1\%` of the stripped-envelope (SESNe) rate, which is itself
       :math:`30.4^{+5.0}_{-4.9}\%` of the total core-collapse rate, so the Type IIb fraction of
       the CC rate is :math:`0.340\times0.304=0.103`. Combined in quadrature with
       :footcite:t:`strolger2015`'s :math:`+27\%/-31\%` normalization uncertainty, this gives
       :attr:`~uvex_transients.transients.supernovae.ShockCoolingIIb.RATE_CI` (see
       :ref:`user_guide_transients_rate_uncertainty`).
   * - Redshift limit
     - :math:`z = 1`
     - --
     - Generous relative to the brief, luminous shock-cooling phase this SED targets.
   * - Duration
     - 20 days
     - --
     - Deliberately short: sampling 20,000 draws from the priors below, Morag+24's own stated
       validity window (its Eqs. 17-18) extends past 10 days for only ~2% of realizations, and
       essentially never reaches the ~15-25 day radioactive-decay peak that a full
       Type IIb light curve would show. Parameter/time combinations outside that window evaluate
       to ``nan`` rather than extrapolating.

SED Model
----------

There are two shock-cooling SED models available, sharing one set of physical parameters (a shock
velocity scale, progenitor radius, envelope/core mass, and an opacity fixed at the
electron-scattering value):

- :class:`~uvex_transients.models.supernovae.IIb.MoragShockCoolingSED` (the default) implements
  Morag+24's full frequency-dependent SED: a blackbody at the photospheric color temperature,
  reshaped by UV line suppression above :math:`\sim3.5\,T_\mathrm{col}` and a separate
  diffusion-limited/free-free prescription below it.
- :class:`~uvex_transients.models.supernovae.IIb.MoragShockCoolingBlackbodySED` implements the
  simpler alternative: a pure blackbody at the same color temperature, with no frequency reshaping.

Both share the same bolometric luminosity and color-temperature evolution, a closed-form,
diffusion-envelope solution calibrated against numerical radiation-hydrodynamics simulations:

.. math::

    L_\mathrm{bol}(t) = L_\mathrm{break}
    \left[\left(\frac{t}{t_\mathrm{break}}\right)^{-4/3}
    + 0.9\,e^{-\sqrt{2t/t_\mathrm{tr}}}\left(\frac{t}{t_\mathrm{break}}\right)^{-0.17}\right],

with the color temperature declining from a scale value near :math:`t_\mathrm{break}` as a broken
power law in time. :math:`L_\mathrm{break}`, :math:`t_\mathrm{break}`, and :math:`t_\mathrm{tr}` are
themselves closed-form combinations of the physical parameters below; see
:footcite:t:`2024MNRAS.528.7137M` for their exact form and for the full frequency-dependent SED
(its Eqs. A1-A12) -- the details are intentionally not reproduced here.

The two models' bolometric luminosities are physically identical, but only
:class:`~uvex_transients.models.supernovae.IIb.MoragShockCoolingBlackbodySED`'s emergent spectrum integrates back to it exactly: a
blackbody conserves its own normalization by construction, while the UV-suppressed reshaping in
:class:`~uvex_transients.models.supernovae.IIb.MoragShockCoolingSED` does not (by as much as ~15-20%, depending on epoch) -- an artifact of
how Morag+24 construct the two pieces independently, not a bug in either implementation.

.. dropdown:: Parameter priors

   All five parameters use fairly broad priors, since the goal here is to sample plausible
   shock-cooling realizations rather than to reproduce a specific calibrating event.

   .. list-table::
      :header-rows: 1
      :widths: 16 12 40 32

      * - Parameter
        - Symbol
        - Prior
        - Notes / Source
      * - ``v_star``
        - :math:`v_*`
        - LogNormal(:math:`v_*/10^{8.5}\,\mathrm{cm\,s^{-1}}`; mean=0, :math:`\sigma`\=0.5)
        - Scale velocity of the shock near the stellar surface.
      * - ``radius``
        - :math:`R`
        - LogNormal(:math:`R/10^{13}\,\mathrm{cm}`; mean=0, :math:`\sigma`\=0.5)
        - Progenitor stellar radius.
      * - ``opacity``
        - :math:`\kappa`
        - Fixed (0.34 :math:`\mathrm{cm^2\,g^{-1}}`)
        - Electron-scattering opacity.
      * - ``envelope_mass``
        - :math:`M_E`
        - LogNormal(:math:`M_E/M_\odot`; mean=0, :math:`\sigma`\=0.5)
        - Envelope mass.
      * - ``core_mass``
        - :math:`M_C`
        - LogNormal(:math:`M_C/M_\odot`; mean=0, :math:`\sigma`\=0.5)
        - Core mass.

Simulated Light Curves
~~~~~~~~~~~~~~~~~~~~~~~

The plot below draws 500 random parameter realizations from the priors above and shows the
resulting bolometric light curves and photospheric temperatures, using the default
:class:`~uvex_transients.models.supernovae.IIb.MoragShockCoolingSED`. No comparison data is overlaid: unlike the other supernova
populations in this package, there is not yet a curated bolometric/temperature dataset for
shock-cooling Type IIb events under ``test_data/transients``.

.. plot::
   :include-source: false

   import numpy as np
   import matplotlib.pyplot as plt
   from astropy import units as u

   from uvex_transients.models.supernovae import MoragShockCoolingSED as SEDClass

   rng = np.random.default_rng(20260910)
   n_samples = 500

   params = SEDClass().sample_parameters(size=n_samples, rng=rng)
   params_grid = {name: value[:, None] for name, value in params.items()}

   t = np.geomspace(0.005, 15, 250) * u.day
   L_bol = SEDClass.eval_bolometric(t, **params_grid)
   T = SEDClass.temperature(t, **params_grid)

   fig, (ax_L, ax_T) = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True)

   for row in range(n_samples):
       ax_L.plot(t.to_value(u.day), L_bol[row].to_value(u.erg / u.s), color="C0", lw=0.4, alpha=0.08)
       ax_T.plot(t.to_value(u.day), T[row].to_value(u.K), color="C3", lw=0.4, alpha=0.08)

   ax_L.set_xscale("log")
   ax_L.set_yscale("log")
   ax_L.set_ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
   ax_L.set_title("Shock-cooling IIb: simulated bolometric light curves (n=500)")

   ax_T.set_yscale("log")
   ax_T.set_xlabel("Time since explosion [days]")
   ax_T.set_ylabel("Photospheric temperature [K]")

   fig.tight_layout()


----

Simulated UVOIR Light Curves (Rubin + UVEX)
---------------------------------------------

Because the shock-cooling phase lasts only days, seeing it well requires a cadence much tighter
than the multi-day cadences typically used for longer-lived transients. The two panels below show
two random parameter realizations at a fixed redshift (:math:`z=0.02`, roughly 90 Mpc), each
observed by Rubin (``g``/``r``/``i``, 30 s visits) and UVEX (``FUV``/``NUV``, 900 s visits) on a
shared 12-hour cadence over the first 12 days: solid curves are the noiseless theory light curves,
points are simulated photometry (shot noise plus, for Rubin, its own photometric-calibration
floor), and open triangles are :math:`\mathrm{SNR}<5` upper limits. This is meant to give a sense
of what the model's SEDs actually look like observationally, not a rate/yield forecast -- see the
other transient pages in this section for that kind of analysis.

.. plot::
   :include-source: false

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


References
-----------

.. footbibliography::
