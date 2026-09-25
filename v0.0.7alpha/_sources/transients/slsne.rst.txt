.. _transients_slsne:

Superluminous Supernovae (SLSNe-I)
=====================================

Superluminous supernovae (SLSNe) reach peak luminosities roughly ten times higher than ordinary
core-collapse or Type Ia supernovae and stay bright for weeks to months. The hydrogen-poor
subclass, SLSNe-I, is the best studied: radioactive :math:`^{56}\mathrm{Ni}` decay alone cannot
supply enough energy to power most of these events, and their bright, blue, slowly-declining
light curves are instead widely explained by the spin-down power of a newly formed,
rapidly-rotating, strongly-magnetized neutron star (a magnetar) embedded in the expanding ejecta
:footcite:p:`2010ApJ...717..245K, 2010ApJ...719L.204W`.

This population is implemented by :class:`~uvex_transients.transients.supernovae.MagnetarSLSNe`,
pairing :class:`~uvex_transients.models.arnett.ArnettMagnetarSpindownSED` with the
rate/duration metadata described below. Unlike the purely phenomenological SED shapes used
elsewhere in this package, this SED is a semi-analytic solution of the underlying diffusion
physics (:footcite:t:`1982ApJ...253..785A`, extended by :footcite:t:`2017ApJ...850...55N` and
:footcite:t:`2015ApJ...799..107W`), so it is grouped with the other detailed physical models.

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
     - :math:`R_\mathrm{CC}(z) = k\,\psi_\mathrm{UV}(z)`; SLSN-I :math:`1/3500` of :math:`R_\mathrm{CC}(z)`
       (:math:`\approx18\ \mathrm{Gpc^{-3}\,yr^{-1}}` locally)
     - :footcite:t:`strolger2015`, :footcite:t:`madau2014`, :footcite:t:`2021MNRAS.500.5142F`
     - :footcite:t:`2021MNRAS.500.5142F` measure a local ratio of SLSN-I to all core-collapse SNe of
       :math:`1/3500^{+2800}_{-720}` (uncertainty on the denominator, i.e. the rate itself spans
       :math:`1/6300` to :math:`1/2780`). Adopted as a constant fraction of the core-collapse rate,
       so it tracks the same star-formation history. Combined in quadrature with
       :footcite:t:`strolger2015`'s :math:`+27\%/-31\%` normalization uncertainty, this gives
       :attr:`~uvex_transients.transients.supernovae.MagnetarSLSNe.RATE_CI` (see
       :ref:`user_guide_transients_rate_uncertainty`).
   * - Redshift limit
     - :math:`z = 4`
     - --
     - Set from an actual ``generate_events``/``filter_by_snr`` run against the default UVEX
       schedule (25 AB mag limiting-magnitude screen, SNR :math:`>5`).
   * - Duration
     - 600 days
     - --
     - Covers the rise (median :math:`\approx30` d) and most of the decline. The bolometric light
       curve falls to :math:`10^{-3}` of peak after a median of :math:`\approx540` d (16th-84th
       percentile 250-1400 d, rest frame), so the slowest events' faint tails are truncated by this
       window rather than fully simulated.

SED Model
----------

*Model Class*: :class:`~uvex_transients.models.arnett.ArnettMagnetarSpindownSED`

The SED model for the SLSNe-I population utilizes the standard Arnett-style
diffusion\ :footcite:p:`2017ApJ...850...55N` model driven by a magnetar spin-down power source.

The magnetar's total rotation energy and spin-down timescale are\ :footcite:p:`2017ApJ...850...55N`

.. math::

    E_\mathrm{mag} = 2.6\times10^{52}\left(\frac{M_\mathrm{NS}}{1.4\,M_\odot}\right)^{3/2}\left(\frac{P}{1\,\mathrm{ms}}\right)^{-2}\ \mathrm{erg},

and

.. math::

    t_\mathrm{mag} = 1.3\times10^{5}\left(\frac{M_\mathrm{NS}}{1.4\,M_\odot}\right)^{3/2}\left(\frac{P}{1\,\mathrm{ms}}\right)^{2}\left(\frac{B_\perp}{10^{14}\,\mathrm{G}}\right)^{-2}\ \mathrm{s}.

We therefore adopt the engine power source

.. math::

    F_\mathrm{mag}(t) = \frac{E_\mathrm{mag}}{t_\mathrm{mag}}\left(1 + \frac{t}{t_\mathrm{mag}}\right)^{-2}.

The resulting bolometric lightcurve, including leakage, is

.. math::

    L(t) = e^{-(t/t_d)^2}\left(1 - e^{-A/t^2}\right)
           \int_0^t 2 F_\mathrm{mag}(t')\,\frac{t'}{t_d}\,e^{(t'/t_d)^2}\,\frac{dt'}{t_d},

where

.. math::

    t_d = \sqrt{\frac{2\kappa M_\mathrm{ej}}{\beta c\,v_\mathrm{ej}}}

is the photon diffusion time and

.. math::

    A = \frac{3\kappa_\gamma M_\mathrm{ej}}{4\pi v_\mathrm{ej}^2},

is the leakage parameter. We adopt the density-profile constant :math:`\beta=13.8`\ :footcite:p:`1982ApJ...253..785A`.
The photosphere expands at the constant ejecta velocity :math:`v_\mathrm{ej}`, so its temperature follows the
Stefan-Boltzmann law until it cools to a floor :math:`T_\mathrm{floor}`, after which it holds there (the photosphere then
recedes rather than continuing to cool):

.. math::

    T(t) = \max\left\{\left[\frac{L(t)}{4\pi\sigma_\mathrm{SB}(v_\mathrm{ej}t)^2}\right]^{1/4},\ T_\mathrm{floor}\right\},
    \qquad
    L_\nu(\nu, t) = L(t)\,\frac{\pi B_\nu(\nu, T(t))}{\sigma_\mathrm{SB}T(t)^4}.


.. list-table::
   :header-rows: 1
   :widths: 16 14 38 32

   * - Parameter
     - Symbol
     - Prior
     - Notes / Source
   * - ``spin_period``
     - :math:`P`
     - TruncatedLogNormal(:math:`\log_{10}(P/\mathrm{ms})`; mean=:math:`\log_{10}(3.0)`,
       :math:`\sigma`\=0.104; bounds :math:`[0.7, 20]` ms)
     - Calibrated as a group with ``B_perp``/``M_ej`` (see note above); bounds are
       :footcite:t:`2017ApJ...850...55N`'s own fit-prior range.
   * - ``B_perp``
     - :math:`B_\perp`
     - TruncatedLogNormal(:math:`\log_{10}(B_\perp/10^{14}\,\mathrm{G})`; mean=:math:`\log_{10}(0.8)`,
       :math:`\sigma`\=0.192; bounds :math:`[0.01, 10]\times10^{14}` G)
     - Calibrated as a group with ``spin_period``/``M_ej`` (see note above); bounds are
       :footcite:t:`2017ApJ...850...55N`'s own fit-prior range.
   * - ``M_ej``
     - :math:`M_\mathrm{ej}`
     - TruncatedLogNormal(:math:`\log_{10}(M_\mathrm{ej}/M_\odot)`; mean=:math:`\log_{10}(4.8)`,
       :math:`\sigma`\=0.152; bounds :math:`[0.1, 100]\,M_\odot`)
     - Calibrated as a group with ``spin_period``/``B_perp`` (see note above); bounds are
       :footcite:t:`2017ApJ...850...55N`'s own fit-prior range.
   * - ``v_ej``
     - :math:`v_\mathrm{ej}`
     - TruncatedNormal(:math:`v_\mathrm{ej}/10^4\,\mathrm{km\,s^{-1}}`; mean=0.9, :math:`\sigma`\=0.3;
       bounds :math:`[0.1, 3.0]`)
     - :math:`\approx9000\ \mathrm{km\,s^{-1}}`, from :math:`\sqrt{2E_K/M_\mathrm{ej}}` at the
       posterior median :math:`E_K=3.9\times10^{51}` erg, :math:`M_\mathrm{ej}=4.8\,M_\odot`
       :footcite:p:`2017ApJ...850...55N`; width chosen here.
   * - ``M_ns``
     - :math:`M_\mathrm{NS}`
     - Uniform(1.4, 2.2) :math:`M_\odot`
     - Fit-prior range :footcite:p:`2017ApJ...850...55N`.
   * - ``kappa``
     - :math:`\kappa`
     - Uniform(0.05, 0.2) :math:`\mathrm{cm^2\,g^{-1}}`
     - Fit-prior range :footcite:p:`2017ApJ...850...55N`.
   * - ``kappa_gamma``
     - :math:`\kappa_\gamma`
     - LogUniform(0.01, 1) :math:`\mathrm{cm^2\,g^{-1}}`
     - Narrower than the fit-prior's full :math:`0.01`-:math:`100\ \mathrm{cm^2\,g^{-1}}` range: only
       a few events have late enough data to constrain :math:`\kappa_\gamma`, but those that do
       favour similarly low values (e.g. SN 2015bn, :math:`\kappa_\gamma\approx0.01`)
       :footcite:p:`2017ApJ...850...55N`.
   * - ``T_floor``
     - :math:`T_\mathrm{floor}`
     - TruncatedNormal(6000 K, :math:`\sigma`\=1000 K; bounds :math:`[3000, 10000]` K)
     - Matches the fit prior exactly :footcite:p:`2017ApJ...850...55N`.

Simulated Light Curves
~~~~~~~~~~~~~~~~~~~~~~~

The plot below draws 300 random parameter realizations from the priors above and shows the
resulting bolometric light curves and photospheric temperatures. Overlaid lightcurves are from
:footcite:t:`gomez2024`, which provides a curated set of SLSN-I light curves from the literature, with bolometric
luminosities derived from multi-band photometry.

.. plot::
   :include-source: false

   import numpy as np
   import matplotlib.pyplot as plt
   from astropy import units as u
   from uvex_transients.utils.lightcurve_archive import LightcurveArchive

   from uvex_transients.models.supernovae import ArnettMagnetarSpindownSED as SEDClass

   rng = np.random.default_rng(20260921)
   n_samples = 300

   params = SEDClass().sample_parameters(size=n_samples, rng=rng)
   params_grid = {name: value[:, None] for name, value in params.items()}

   t = np.geomspace(1, 1500, 400) * u.day
   L_bol = SEDClass.eval_bolometric(t, **params_grid)
   T = SEDClass.temperature(t, **params_grid)

   fig, (ax_L, ax_T) = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True)

   for row in range(n_samples):
       ax_L.plot(t.to_value(u.day), L_bol[row].to_value(u.erg / u.s), color="C0", lw=0.4, alpha=0.15)
       ax_T.plot(t.to_value(u.day), T[row].to_value(u.K), color="C3", lw=0.4, alpha=0.15)

   # Add the archival sources.
   archive = LightcurveArchive()
   events = archive.events("supernovae/SLSN-I")

   n_temp_events = sum(1 for name in events if "T_phot" in archive.fields("supernovae/SLSN-I", name))
   temp_plotted = 0

   for i, name in enumerate(events):
       lbol = archive.table("supernovae/SLSN-I", name, "L_bol")
       ax_L.plot(
           lbol["time"].to_value(u.day), lbol["L_bol"].to_value(u.erg / u.s),
           color="k", lw=0.5, alpha=0.25,
           label=f"Gomez+2024 (n={len(events)})" if i == 0 else None,
       )

       if "T_phot" in archive.fields("supernovae/SLSN-I", name):
           t_phot = archive.table("supernovae/SLSN-I", name, "T_phot")
           ax_T.plot(
               t_phot["time"].to_value(u.day), t_phot["T_phot"].to_value(u.K),
               color="k", lw=0.5, alpha=0.25,
               label=f"Gomez+2024 (n={n_temp_events})" if temp_plotted == 0 else None,
           )
           temp_plotted += 1

   ax_L.set_xscale("log")
   ax_L.set_yscale("log")
   ax_L.set_ylim([1e41, 1e46])
   ax_L.set_ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
   ax_L.set_title(f"SLSN-I: simulated bolometric light curves (n={n_samples})")
   ax_L.legend(loc="upper right", fontsize=8, frameon=False)

   ax_T.set_xscale("log")
   ax_T.set_yscale("log")
   ax_T.set_xlabel("Time since explosion [days]")
   ax_T.set_ylabel("Photospheric temperature [K]")
   ax_T.set_title(f"SLSN-I: simulated photospheric temperatures (n={n_samples})")
   ax_T.legend(loc="upper right", fontsize=8, frameon=False)

   fig.tight_layout()


----

Observability Summary
----------------------

The light curve here has no single parameter that is exactly the bolometric peak time (:math:`L(t)`
is a numerically integrated diffusion solution), so each event's peak apparent magnitude is found by
searching a rest-frame time grid rather than evaluating at one closed-form epoch. Below are the
redshifts :math:`z` and corresponding bandpass peak apparent AB magnitudes :math:`m_\mathrm{AB}` of
2000 simulated events drawn from the priors above, with the UVEX 1 Dwell limit of :math:`m<24.5`
overplotted. This justifies the :math:`z=4` redshift limit adopted above.

.. plot::
   :include-source: false

   import numpy as np
   import matplotlib.pyplot as plt
   from astropy import units as u
   from scipy.stats import gaussian_kde

   from m4opt.missions import uvex
   from uvex_transients.transients.supernovae import MagnetarSLSNe

   rng = np.random.default_rng(20260921)
   n_samples = 2000

   slsne = MagnetarSLSNe()
   z = slsne.sample_event_redshift(n_samples, rng=rng)
   params = slsne.sed.sample_parameters(size=n_samples, rng=rng)
   params_grid = {name: value[:, None] for name, value in params.items()}

   t_grid_rest = np.geomspace(1, 600, 250) * u.day
   t_obs_grid = t_grid_rest[None, :] * (1.0 + z)[:, None]
   z_grid_bcast = np.broadcast_to(z[:, None], t_obs_grid.shape)

   bandpasses = uvex.detector.bandpasses
   band_names = list(bandpasses)

   fig, axes = plt.subplots(1, len(band_names), figsize=(10.5, 4.8), sharey=True)

   for ax, band_name in zip(axes, band_names):
       mag_curve = slsne.sed.mag_bandpass(
           bandpasses[band_name], t_obs_grid, redshift=z_grid_bcast, **params_grid
       ).to_value(u.ABmag)
       mag = np.nanmin(mag_curve, axis=1)
       finite = np.isfinite(mag)
       z_finite, mag_finite = z[finite], mag[finite]

       ax.scatter(z_finite, mag_finite, s=5, ec="k", fc="k", alpha=0.4, label="Simulated events")

       kde = gaussian_kde(np.vstack([z_finite, mag_finite]))
       z_kde = np.linspace(z_finite.min(), z_finite.max(), 150)
       mag_kde = np.linspace(mag_finite.min(), mag_finite.max(), 150)
       Z_grid, Mag_grid = np.meshgrid(z_kde, mag_kde)
       density = kde(np.vstack([Z_grid.ravel(), Mag_grid.ravel()])).reshape(Z_grid.shape)
       ax.contour(Z_grid, Mag_grid, density, levels=6, colors="k", linewidths=0.7)

       ax.axhline(24.5, color="firebrick", ls="--", lw=1.2, label="UVEX limit (1 Dwell)")

       ax.invert_yaxis()
       ax.set_xlabel("Redshift")
       ax.set_title(f"UVEX {band_name}")
       ax.legend(loc="upper right", fontsize=8, frameon=False)
       ax.set_ylim([32, 16])

   axes[0].set_ylabel("Peak apparent AB magnitude")
   fig.suptitle(f"SLSN-I: peak apparent magnitude vs. redshift (n={n_samples})")
   fig.tight_layout()

The anticipated rate of SLSNe-I detectable by UVEX at this limit is as follows, assuming that any
event above the :math:`m<24.5` limit is detectable, and that the population is isotropic and
homogeneous in comoving volume out to :math:`z=4`:

.. plot::
   :include-source: false

   import numpy as np
   import matplotlib.pyplot as plt
   from astropy import units as u

   from m4opt.missions import uvex
   from uvex_transients.transients.supernovae import MagnetarSLSNe
   from uvex_transients.utils.plotting import add_funnel_legend, get_band_color, plot_rate_bars

   rng = np.random.default_rng(20260921)
   n_samples = 2000

   slsne = MagnetarSLSNe()
   redshift = slsne.sample_event_redshift(n_samples, rng=rng)
   params = slsne.sed.sample_parameters(size=n_samples, rng=rng)
   params_grid = {name: value[:, None] for name, value in params.items()}

   # The all-sky rate, with no survey footprint applied.
   all_sky_rate = slsne.all_sky_rate

   t_grid_rest = np.geomspace(1, 600, 250) * u.day
   t_obs_grid = t_grid_rest[None, :] * (1.0 + redshift)[:, None]
   z_grid_bcast = np.broadcast_to(redshift[:, None], t_obs_grid.shape)

   visible_counts = {}
   for band_name, bandpass in uvex.detector.bandpasses.items():
       mag_curve = slsne.sed.mag_bandpass(
           bandpass, t_obs_grid, redshift=z_grid_bcast, **params_grid
       ).to_value(u.ABmag)
       magnitudes = np.nanmin(mag_curve, axis=1)

       visible = magnitudes < 24.5
       visible_counts[band_name] = int(np.count_nonzero(visible))

       print(
           f"{band_name}: {visible_counts[band_name] / n_samples * all_sky_rate:.2f} "
           f"({visible_counts[band_name] / n_samples:.1%} of events visible)"
       )

   # Plot all-sky visible rates, with MC (statistical) and rate (systematic) uncertainty --
   # see uvex_transients.utils.plotting.plot_rate_bars.
   band_names = list(visible_counts)

   fig, ax = plt.subplots(figsize=(5, 4))
   plot_rate_bars(
       ax,
       band_names,
       [visible_counts[band] for band in band_names],
       n_samples,
       all_sky_rate,
       rate_ci=slsne.RATE_CI,
       color=[get_band_color(band) for band in band_names],
   )
   ax.set_yscale("log")
   ax.set_ylabel(r"All-sky rate [yr$^{-1}$]")
   ax.set_title("Peak-visible SLSN-I rate")
   add_funnel_legend(ax, loc="lower right")

   fig.tight_layout()

References
-----------

.. footbibliography::
