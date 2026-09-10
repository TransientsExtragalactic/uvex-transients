.. _transients_tdes:

Tidal Disruption Events
=========================

A tidal disruption event (TDE) occurs when a star passes close enough to a (typically
super-massive) black hole that the hole's tidal field exceeds the star's self-gravity, unbinding
and disrupting it. Roughly half of the stellar debris remains bound and eventually falls back onto
the black hole, powering a luminous, months-long flare. Optically/UV-selected TDEs -- the class
UVEX is sensitive to -- are observed to radiate as blue, roughly constant-temperature thermal
sources with characteristic blackbody temperatures of a few :math:`\times10^4` K
:footcite:p:`2021ApJ...908....4V`, in contrast to the reprocessing-independent picture originally
expected from a purely X-ray-bright accretion disk. :footcite:t:`2021ApJ...908....4V` present the
largest homogeneously-analyzed optical/UV TDE sample to date (17 new ZTF-discovered events plus 22
from the literature) and provide the empirical rise/decline/temperature statistics this SED's
default priors are built from. This population is implemented by
:class:`~uvex_transients.transients.TDEs.TidalDisruptionEvent`, pairing
:class:`~uvex_transients.models.tdes.alush_stone.AlushStoneTDESED` with the rate/duration metadata described
below.

Transient Rates
----------------

The adopted volumetric rate, :math:`3.1\times10^{-7}\ \mathrm{Mpc^{-3}\,yr^{-1}}`, is taken from
the maximum-volume-corrected demographic analysis of :footcite:t:`yao2023`, using 33
spectroscopically-confirmed TDEs from three years of the Zwicky Transient Facility. The rate is
adopted as constant with redshift: its evolution with redshift remains a subject of active debate
-- :footcite:t:`karmen2026` show the observed redshift-dependent TDE rate is highly sensitive to
the poorly-constrained evolution of the supermassive black hole mass function itself -- so no
evolution is modeled here. A redshift limit of :math:`z=1` and a 200-day duration window are used,
both generous relative to the timescales and luminosities of the observed optical/UV TDE
population.

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - Rate
     - Redshift Limit
     - Duration
     - Source
   * - :math:`3.1\times10^{-7}\ \mathrm{Mpc^{-3}\,yr^{-1}}` (constant in :math:`z`)
     - :math:`z = 1`
     - 200 days
     - :footcite:t:`yao2023`

SED Model
----------

:class:`~uvex_transients.models.tdes.alush_stone.AlushStoneTDESED` extends the simple constant-temperature
picture (implemented separately in
:class:`~uvex_transients.models.tdes.van_velzen.VanVelzenTDESED`, following the
Gaussian-rise/exponential-decline parameterization of :footcite:t:`2021ApJ...908....4V`) with a
second, late-time component: a magnetized accretion-disk plateau. The early-time photospheric
emission is exactly :class:`~uvex_transients.models.tdes.van_velzen.VanVelzenTDESED`'s own
Gaussian-rise/exponential-decline light curve
(:class:`~uvex_transients.models.lightcurves.generic.GREDLightcurve`) times a constant-temperature
blackbody, while the late-time disk plateau contributes its own, separately-normalized blackbody
component that smoothly softens from a flat plateau to a power-law decline:

.. math::

    L_\nu(\nu, t) = L_\mathrm{bol}^\mathrm{early}(t)\, S(\nu, T)
        + L_\mathrm{bol}^\mathrm{plat}(t)\, S(\nu, T_\mathrm{p}),
    \qquad
    L_\mathrm{bol}^\mathrm{plat}(t) = L_\mathrm{p}
        \left(1 + \frac{t - t_\mathrm{peak}}{\tau_\mathrm{p}}\right)^{-\alpha_\mathrm{p}},

where :math:`S(\nu, T)` is the normalized blackbody shape and :math:`t_\mathrm{peak}=5\sigma` is
the early component's own peak time. The two components are summed in linear luminosity space, not
switched between, since a real disk plateau does not sharply replace the fading early-time
emission. The plateau's functional form and its default decline index,
:math:`\alpha_\mathrm{p}\sim5/6`, follow the magnetized-disk model of
:footcite:t:`2025arXiv250303811A`, which predicts an :math:`L_\mathrm{UV}\propto t^{-5/6}`
late-time decline persisting for decades to centuries; :footcite:t:`2025arXiv251024696A` fit this
same family of theory-agnostic plateau shapes to real late-time TDE light curves and find that
roughly a third of their sample shows a statistically significant, evolving (rather than perfectly
flat) plateau, motivating leaving :math:`\alpha_\mathrm{p}` a free parameter here rather than fixing
it at the theoretical value. The early-time parameters (``temperature``, ``sigma_rise``,
``tau_decline``) reuse the same :footcite:t:`2021ApJ...908....4V`-informed defaults as
:class:`~uvex_transients.models.tdes.van_velzen.VanVelzenTDESED`; the plateau parameters do not yet have
literature-calibrated priors and use order-of-magnitude fiducial scales instead.

.. list-table::
   :header-rows: 1
   :widths: 16 12 40 32

   * - Parameter
     - Symbol
     - Prior
     - Notes / Source
   * - ``amplitude``
     - :math:`L_0`
     - LogNormal(:math:`\log_{10}(L_0/\mathrm{erg\,s^{-1}})`; mean=44.0, :math:`\sigma`\=0.1)
     - Early-time peak bolometric luminosity :footcite:p:`2021ApJ...908....4V`.
   * - ``temperature``
     - :math:`T`
     - LogNormal(:math:`\log_{10}(T/\mathrm{K})`; mean=4.3, :math:`\sigma`\=0.1)
     - Early-time photospheric temperature, :math:`\approx2\times10^4` K :footcite:p:`2021ApJ...908....4V`.
   * - ``sigma_rise``
     - :math:`\sigma`
     - LogNormal(:math:`\log_{10}(\sigma/\mathrm{d})`; mean=0.91, :math:`\sigma`\=0.2)
     - Gaussian width of the pre-peak rise :footcite:p:`2021ApJ...908....4V`.
   * - ``tau_decline``
     - :math:`\tau`
     - LogNormal(:math:`\log_{10}(\tau/\mathrm{d})`; mean=1.8, :math:`\sigma`\=0.2)
     - Exponential decline timescale after peak :footcite:p:`2021ApJ...908....4V`.
   * - ``plateau_temperature``
     - :math:`T_\mathrm{p}`
     - LogNormal(:math:`\log_{10}(T_\mathrm{p}/\mathrm{K})`; mean=4.0, :math:`\sigma`\=0.3)
     - Late-time disk-plateau blackbody temperature; fiducial scale, not yet calibrated.
   * - ``plateau_amplitude``
     - :math:`L_\mathrm{p}`
     - LogNormal(:math:`\log_{10}(L_\mathrm{p}/\mathrm{erg\,s^{-1}})`; mean=41.5, :math:`\sigma`\=0.2)
     - Plateau bolometric luminosity at :math:`t_\mathrm{peak}`; fiducial scale, not yet calibrated.
   * - ``plateau_timescale``
     - :math:`\tau_\mathrm{p}`
     - LogNormal(:math:`\log_{10}(\tau_\mathrm{p}/\mathrm{d})`; mean=2.3, :math:`\sigma`\=0.3)
     - Timescale over which the plateau softens into its power-law decline.
   * - ``plateau_decline``
     - :math:`\alpha_\mathrm{p}`
     - Uniform(0, 2)
     - Late-time decline index, centered near the magnetized-disk prediction :math:`5/6`
       :footcite:p:`2025arXiv250303811A,2025arXiv251024696A`.

Simulated Light Curves
~~~~~~~~~~~~~~~~~~~~~~~

Because the model sums two independently-normalized blackbody components, its total bolometric
luminosity is exactly the sum of each component's own (exactly known) bolometric luminosity, and
its luminosity-weighted effective temperature smoothly transitions from the early photospheric
temperature to the plateau temperature as the disk plateau takes over. The plot below draws 1000
random parameter realizations from the priors above and shows the resulting bolometric light
curves and effective temperatures.

.. plot::
   :include-source: false

   import numpy as np
   import matplotlib.pyplot as plt
   from astropy import units as u

   from uvex_transients.models.tdes import AlushStoneTDESED as SEDClass
   from uvex_transients.models.lightcurves.generic import GREDLightcurve

   rng = np.random.default_rng(20260910)
   n_samples = 1000

   params = SEDClass().sample_parameters(size=n_samples, rng=rng)
   params_grid = {name: value[:, None] for name, value in params.items()}

   t = np.geomspace(0.5, 200, 200) * u.day

   # Each branch is an independently-normalized blackbody, so the total bolometric
   # luminosity is exactly the sum of the two components' own bolometric luminosities
   # (see AlushStoneTDESED's docstring for the closed-form plateau branch).
   t_peak = 5 * params_grid["sigma_rise"]
   early_amp = GREDLightcurve.eval(
       t,
       amplitude=params_grid["amplitude"],
       sigma_rise=params_grid["sigma_rise"],
       tau_decline=params_grid["tau_decline"],
   )

   x = ((t - t_peak) / params_grid["plateau_timescale"]).to_value(u.dimensionless_unscaled)
   alpha_p = params_grid["plateau_decline"].to_value(u.dimensionless_unscaled)
   plateau_active = x >= 0.0
   late_amp = (
       np.where(
           plateau_active,
           params_grid["plateau_amplitude"].to_value(u.erg / u.s) * (1.0 + np.where(plateau_active, x, 0.0)) ** (-alpha_p),
           0.0,
       )
       * (u.erg / u.s)
   )

   L_bol = early_amp + late_amp
   T_eff = (early_amp * params_grid["temperature"] + late_amp * params_grid["plateau_temperature"]) / L_bol

   fig, (ax_L, ax_T) = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True)

   for row in range(n_samples):
       ax_L.plot(t.to_value(u.day), L_bol[row].to_value(u.erg / u.s), color="C0", lw=0.4, alpha=0.06)
       ax_T.plot(t.to_value(u.day), T_eff[row].to_value(u.K), color="C3", lw=0.4, alpha=0.06)

   ax_L.set_xscale("log")
   ax_L.set_yscale("log")
   ax_L.set_ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
   ax_L.set_title("Tidal disruption events: simulated bolometric light curves (n=1000)")

   ax_T.set_yscale("log")
   ax_T.set_xlabel("Time since disruption [days]")
   ax_T.set_ylabel("Luminosity-weighted effective temperature [K]")

   fig.tight_layout()

References
-----------

.. footbibliography::
