.. _transients_type_i:

Type I (Stripped-Envelope) Supernovae
=======================================

Type Ib and Type Ic supernovae are core-collapse explosions of massive stars that lost their
hydrogen envelope (Ib: helium remains; Ic: helium is also stripped) before exploding. Their light
curves are powered by the radioactive decay of :math:`^{56}\mathrm{Ni}` and are typically a single
peak, roughly two to three weeks after explosion, followed by a decline; unlike Type IIb SNe
(:ref:`transients_type_ii`) they lack a distinct early shock-cooling peak. All three subtypes on
this page -- Ib, Ic and Ic-BL (broad-lined) -- are modeled with the same first-principles
Arnett-style radioactive-decay diffusion physics as Type Ia
(:class:`~uvex_transients.models.arnett.ArnettDecaySED`), differing only in their priors (and, for
Ic-BL, in its higher characteristic ejecta velocities and kinetic energies); the three populations
are implemented as sibling transient classes below.

.. note::

   The Type Ib and Type Ic priors on ``M_Ni``, ``M_ej`` and ``v_ej`` are the per-subtype sample
   statistics (mean, sample standard deviation) of the analytical Arnett-model fits in Table 6 of
   :footcite:t:`lyman2016` (13 Ib and 8 Ic events). ``kappa`` is fixed at the single value
   (:math:`0.06\,\mathrm{cm^2\,g^{-1}}`) Lyman et al. 2016 themselves assume for every fit in their
   sample rather than fitting per event. ``kappa_gamma`` has no analogue in Lyman et al. 2016's
   leakage-free formalism (the original Arnett 1982 diffusion model), so it is instead fixed at
   :math:`0.04\,\mathrm{cm^2\,g^{-1}}`, comparable to `Type Ia`'s Scalzo+14-derived value. Ib and Ic
   share this same construction and differ only in the subsample statistics and event rates. Type
   Ic-BL instead uses sample statistics of its own explosion-property table -- see its tab below.

.. tab-set::

   .. tab-item:: Type Ia

      Type Ia supernovae are thermonuclear explosions of carbon-oxygen white dwarfs, not
      core-collapse events -- there is no compact remnant and no massive-star progenitor, so
      neither this population's SED nor its rate shares any machinery with the Type Ib/Ic/II
      populations elsewhere on this page. Their light curves are powered by the radioactive decay
      of :math:`^{56}\mathrm{Ni}` synthesized in the explosion, following the same Arnett-style
      diffusion physics as the SLSNe-I model (:ref:`transients_slsne`), but without a magnetar
      central engine.

      Implemented by :class:`~uvex_transients.transients.supernovae.TypeIaSNe`, pairing
      :class:`~uvex_transients.models.supernovae.Ia.TypeIaSED` with a delay-time-distribution
      rate (below) rather than a fixed fraction of the core-collapse rate.

      .. rubric:: Quick Facts

      .. list-table::
         :header-rows: 1
         :widths: 15 25 15 45

         * - Quantity
           - Value
           - Source
           - Notes
         * - Rate
           - Cosmic star formation history convolved with a power-law delay-time distribution
             (DTD); local rate :math:`\approx2.3\times10^{-5}\ \mathrm{Mpc^{-3}\,yr^{-1}}`
             (:math:`\approx2.3\times10^{4}\ \mathrm{Gpc^{-3}\,yr^{-1}}`)
           - :footcite:t:`2017ApJ...848...25M`, :footcite:t:`madau2014`
           - :func:`~uvex_transients.utils.cosmology.supernovae_Ia_rate`: DTD
             :math:`\Psi(\tau)\propto\tau^{-1.1}` for :math:`\tau>40` Myr, normalized to
             :math:`N_\mathrm{Ia}/M_\star=(1.3\pm0.1)\times10^{-3}\,M_\odot^{-1}` (Maoz & Graur
             2017), convolved with the Madau & Dickinson (2014) star formation history. Unlike the
             core-collapse subtypes' instantaneous tracer, the broadly distributed delay times
             give SNe Ia a flatter, slower-declining rate shape at high redshift. The rate is
             exactly linear in :math:`N_\mathrm{Ia}/M_\star`, so
             :attr:`~uvex_transients.transients.supernovae.TypeIaSNe.RATE_CI` (:math:`\pm7.7\%`)
             is evaluated by calling `supernovae_Ia_rate` directly at that normalization's
             :math:`\pm1\sigma` endpoints, rather than assumed analytically -- see
             :ref:`user_guide_transients_rate_uncertainty`.
         * - Redshift limit
           - :math:`z = 1`
           - --
           - Set from an actual ``sample_event_redshift``/peak-apparent-magnitude check against
             the UVEX bandpasses (25 AB mag limiting-magnitude screen): no simulated event peaks
             above the limit beyond :math:`z\approx0.8` in either band, and the NUV-detected
             fraction per redshift bin has already fallen to zero by :math:`z=1`.
         * - Duration
           - 365 days
           - --
           - Covers the rise to peak (median :math:`\approx14` d after explosion) through the
             decline to :math:`10^{-3}` of peak for nearly the whole prior (16th-84th percentile
             :math:`\approx270`-:math:`325` d, rest frame).

      .. rubric:: SED Model

      *Model Class*: :class:`~uvex_transients.models.supernovae.Ia.TypeIaSED`

      :class:`~uvex_transients.models.supernovae.Ia.TypeIaSED` reuses
      :class:`~uvex_transients.models.arnett.ArnettDecaySED`'s Arnett-style radioactive-decay
      diffusion light curve and floored-photosphere blackbody entirely (the same
      :math:`L(t)`/:math:`T(t)` machinery documented for the SLSNe-I model in
      :ref:`transients_slsne`, but driven by :math:`^{56}\mathrm{Ni}\to{}^{56}\mathrm{Co}\to{}^{56}\mathrm{Fe}`
      decay heating rather than magnetar spin-down):

      .. math::

          F_\mathrm{decay}(t) = M_\mathrm{Ni}\left[\epsilon_\mathrm{Ni}\,e^{-t/\tau_\mathrm{Ni}}
          + \epsilon_\mathrm{Co}\left(e^{-t/\tau_\mathrm{Co}} - e^{-t/\tau_\mathrm{Ni}}\right)\right],
          \qquad
          T(t) = \max\left\{\left[\frac{L(t)}{4\pi\sigma_\mathrm{SB}(v_\mathrm{ej}t)^2}\right]^{1/4},
          T_\mathrm{floor}\right\}.

      The priors follow :footcite:t:`sarin2026` (a sample of 2205 SNe Ia from ZTF): ``M_Ni`` and
      ``M_ej`` are the population-level Gaussians from that paper's hierarchical Arnett-model fit
      (:math:`\mu_\mathrm{Ni}=0.64\,M_\odot`, :math:`\sigma_\mathrm{Ni}=0.42\,M_\odot`;
      :math:`\mu_\mathrm{ej}=1.26\,M_\odot`, :math:`\sigma_\mathrm{ej}=0.33\,M_\odot`),
      ``kappa_gamma`` is fixed at the value adopted by :footcite:t:`scalzo2014`, and ``kappa`` is
      uniform over that paper's marginalization range.

      .. dropdown:: Parameter priors

         .. list-table::
            :header-rows: 1
            :widths: 16 14 38 32

            * - Parameter
              - Symbol
              - Prior
              - Notes / Source
            * - ``M_Ni``
              - :math:`M_\mathrm{Ni}`
              - TruncatedNormal(0.64, :math:`\sigma`\=0.42; bounds :math:`[0.05, 3.0]\,M_\odot`)
              - :footcite:t:`sarin2026`.
            * - ``M_ej``
              - :math:`M_\mathrm{ej}`
              - TruncatedNormal(1.26, :math:`\sigma`\=0.33; bounds :math:`[0.05, 3.0]\,M_\odot`)
              - :footcite:t:`sarin2026`.
            * - ``v_ej``
              - :math:`v_\mathrm{ej}`
              - Normal(11, :math:`\sigma`\=1) :math:`\times10^3\ \mathrm{km\,s^{-1}}`
              - :footcite:t:`sarin2026`.
            * - ``kappa``
              - :math:`\kappa`
              - Uniform(0.05, 0.15) :math:`\mathrm{cm^2\,g^{-1}}`
              - Marginalization-prior range, :footcite:t:`sarin2026`.
            * - ``kappa_gamma``
              - :math:`\kappa_\gamma`
              - Fixed, 0.03 :math:`\mathrm{cm^2\,g^{-1}}`
              - :footcite:t:`scalzo2014`.
            * - ``T_floor``
              - :math:`T_\mathrm{floor}`
              - TruncatedNormal(6000 K, :math:`\sigma`\=1000 K; bounds :math:`[3000, 10000]` K)
              - Same floor as the other Arnett-based models on this site (see
                :ref:`transients_slsne`); not calibrated against SNe Ia data specifically.

      .. rubric:: Simulated Light Curves

      The plot below draws 300 random parameter realizations from the priors above and shows the
      resulting bolometric light curves and photospheric temperatures. The top panel is overlaid
      with the individual bolometric light curves (time since explosion) of the five Type Ia SNe
      of :footcite:t:`sharon2025`. No photospheric temperature data is bundled for Type Ia, so the
      bottom panel is unadorned.

      .. plot::
         :include-source: false

         import numpy as np
         import matplotlib.pyplot as plt
         from astropy import units as u

         from uvex_transients.models.supernovae import TypeIaSED as SEDClass
         from uvex_transients.utils.lightcurve_archive import LightcurveArchive

         rng = np.random.default_rng(20260923)
         n_samples = 300

         params = SEDClass().sample_parameters(size=n_samples, rng=rng)
         params_grid = {name: value[:, None] for name, value in params.items()}
         archive = LightcurveArchive()

         t = np.geomspace(0.5, 365, 400) * u.day
         L_bol = SEDClass.eval_bolometric(t, **params_grid)
         T = SEDClass.temperature(t, **params_grid)

         fig, (ax_L, ax_T) = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True)

         for row in range(n_samples):
             ax_L.plot(t.to_value(u.day), L_bol[row].to_value(u.erg / u.s), color="C0", lw=0.4, alpha=0.15)
             ax_T.plot(t.to_value(u.day), T[row].to_value(u.K), color="C3", lw=0.4, alpha=0.15)

         sharon_events = archive.events("supernovae/Ia")
         for i, name in enumerate(sharon_events):
             lbol_obs = archive.table("supernovae/Ia", name, "L_bol")
             ax_L.plot(
                 lbol_obs["time"].to_value(u.day), lbol_obs["L_bol"].to_value(u.erg / u.s),
                 color="k", lw=0.8, marker="o", ms=2.5, alpha=0.7,
                 label="Sharon+25 (n=%d)" % len(sharon_events) if i == 0 else None,
             )

         ax_L.set_xscale("log")
         ax_L.set_yscale("log")
         ax_L.set_ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
         ax_L.set_title(f"Type Ia: simulated bolometric light curves (n={n_samples})")
         ax_L.legend(loc="lower left", fontsize=8, frameon=False)

         ax_T.set_xscale("log")
         ax_T.set_yscale("log")
         ax_T.set_xlabel("Time since explosion [days]")
         ax_T.set_ylabel("Photospheric temperature [K]")
         ax_T.set_title(f"Type Ia: simulated photospheric temperatures (n={n_samples})")

         fig.tight_layout()

      .. rubric:: Observability Summary

      Below are the redshifts :math:`z` and corresponding bandpass peak apparent AB magnitudes
      :math:`m_\mathrm{AB}` of 2000 simulated events drawn from the priors above (with
      `redshift_limit` temporarily raised to 4 to show the falloff), with the UVEX 1 Dwell limit
      of :math:`m<24.5` overplotted. This justifies the :math:`z=1` redshift limit adopted above.

      .. plot::
         :include-source: false

         import numpy as np
         import matplotlib.pyplot as plt
         from astropy import units as u

         from m4opt.missions import uvex
         from uvex_transients.transients.supernovae import TypeIaSNe

         rng = np.random.default_rng(20260923)
         n_samples = 2000

         sn = TypeIaSNe()
         sn.redshift_limit = 4.0
         z = sn.sample_event_redshift(n_samples, rng=rng)
         params = sn.sed.sample_parameters(size=n_samples, rng=rng)
         params_grid = {name: value[:, None] for name, value in params.items()}

         t_grid_rest = np.geomspace(0.1, 365, 300) * u.day
         t_obs_grid = t_grid_rest[None, :] * (1.0 + z)[:, None]
         z_grid_bcast = np.broadcast_to(z[:, None], t_obs_grid.shape)

         bandpasses = uvex.detector.bandpasses
         band_names = list(bandpasses)

         fig, axes = plt.subplots(1, len(band_names), figsize=(10.5, 4.8), sharey=True)

         for ax, band_name in zip(axes, band_names):
             mag_curve = sn.sed.mag_bandpass(
                 bandpasses[band_name], t_obs_grid, redshift=z_grid_bcast, **params_grid
             ).to_value(u.ABmag)
             mag = np.nanmin(mag_curve, axis=1)
             finite = np.isfinite(mag)

             ax.scatter(z[finite], mag[finite], s=5, ec="k", fc="k", alpha=0.5, label="Simulated events")
             ax.axhline(24.5, color="firebrick", ls="--", lw=1.2, label="UVEX limit (1 Dwell)")

             ax.invert_yaxis()
             ax.set_xlabel("Redshift")
             ax.set_title(f"UVEX {band_name}")
             ax.legend(loc="upper right", fontsize=8, frameon=False)
             ax.set_ylim([32, 16])

         axes[0].set_ylabel("Peak apparent AB magnitude")
         fig.suptitle(f"Type Ia: peak apparent magnitude vs. redshift (n={n_samples})")
         fig.tight_layout()

      The anticipated rate of SNe Ia detectable by UVEX at this limit is as follows, assuming
      that any event above the :math:`m<24.5` limit is detectable, and that the population is
      isotropic and homogeneous in comoving volume out to :math:`z=1`:

      .. plot::
         :include-source: false

         import numpy as np
         import matplotlib.pyplot as plt
         from astropy import units as u

         from m4opt.missions import uvex
         from uvex_transients.transients.supernovae import TypeIaSNe
         from uvex_transients.utils.plotting import add_funnel_legend, get_band_color, plot_rate_bars

         rng = np.random.default_rng(20260923)
         n_samples = 2000

         sn = TypeIaSNe()
         redshift = sn.sample_event_redshift(n_samples, rng=rng)
         params = sn.sed.sample_parameters(size=n_samples, rng=rng)
         params_grid = {pname: value[:, None] for pname, value in params.items()}

         # The all-sky rate, with no survey footprint applied.
         all_sky_rate = sn.all_sky_rate

         t_grid_rest = np.geomspace(0.1, 365, 300) * u.day
         t_obs_grid = t_grid_rest[None, :] * (1.0 + redshift)[:, None]
         z_grid_bcast = np.broadcast_to(redshift[:, None], t_obs_grid.shape)

         visible_counts = {}
         for band_name, bandpass in uvex.detector.bandpasses.items():
             mag_curve = sn.sed.mag_bandpass(
                 bandpass, t_obs_grid, redshift=z_grid_bcast, **params_grid
             ).to_value(u.ABmag)
             magnitudes = np.nanmin(mag_curve, axis=1)

             visible = magnitudes < 24.5
             visible_counts[band_name] = int(np.count_nonzero(visible))

             print(
                 f"{band_name}: {visible_counts[band_name] / n_samples * all_sky_rate:.2f} "
                 f"({visible_counts[band_name] / n_samples:.1%} of events visible)"
             )

         # Plot all-sky visible rates, with MC (statistical) and rate (systematic)
         # uncertainty -- see uvex_transients.utils.plotting.plot_rate_bars.
         band_names = list(visible_counts)

         fig, ax = plt.subplots(figsize=(5, 4))
         plot_rate_bars(
             ax,
             band_names,
             [visible_counts[band] for band in band_names],
             n_samples,
             all_sky_rate,
             rate_ci=sn.RATE_CI,
             color=[get_band_color(band) for band in band_names],
         )
         ax.set_yscale("log")
         ax.set_ylabel(r"All-sky rate [yr$^{-1}$]")
         ax.set_title("Peak-visible Type Ia rate")
         add_funnel_legend(ax, loc="lower right")

         fig.tight_layout()

   .. tab-item:: Type Ib

      Implemented by :class:`~uvex_transients.transients.supernovae.TypeIbSNe`, pairing
      :class:`~uvex_transients.models.supernovae.Ibc.TypeIbSED` with the rate/duration metadata
      described below.

      .. rubric:: Quick Facts

      .. list-table::
         :header-rows: 1
         :widths: 15 25 15 45

         * - Quantity
           - Value
           - Source
           - Notes
         * - Rate
           - :math:`R_\mathrm{CC}(z) = k\,\psi_\mathrm{UV}(z)`; Type Ib 10.8% of :math:`R_\mathrm{CC}(z)`
           - :footcite:t:`strolger2015`, :footcite:t:`madau2014`,
             :footcite:t:`shivvers2017`
           - Tracks the cosmic star-formation history. :footcite:t:`shivvers2017` find Ib is
             :math:`35.6\pm11.4\%` of the stripped-envelope (SESNe) rate, which is itself
             :math:`30.4^{+5.0}_{-4.9}\%` of the total core-collapse rate, so the Type Ib fraction
             is :math:`0.356\times0.304=0.108`. Combined in quadrature with
             :footcite:t:`strolger2015`'s :math:`+27\%/-31\%` normalization uncertainty, this gives
             :attr:`~uvex_transients.transients.supernovae.TypeIbSNe.RATE_CI` (see
             :ref:`user_guide_transients_rate_uncertainty`).
         * - Redshift limit
           - :math:`z = 0.5`
           - --
           - Below the Type IIP and IIb limits, since these events peak at lower luminosity and are cool
             in the UV: in the observability check below, no simulated event beyond
             :math:`z \approx 0.3` clears the UVEX limit.
         * - Duration
           - 100 days
           - --
           - Covers the rise, peak (:math:`t_p \approx 16` d after explosion, prior median) and the decline.

      .. rubric:: SED Model

      *Model Class*: :class:`~uvex_transients.models.supernovae.Ibc.TypeIbSED`

      :class:`~uvex_transients.models.supernovae.Ibc.TypeIbSED` reuses
      :class:`~uvex_transients.models.arnett.ArnettDecaySED`'s Arnett-style radioactive-decay
      diffusion light curve and floored-photosphere blackbody entirely -- the same
      :math:`L(t)`/:math:`T(t)` machinery documented for `Type Ia` above -- with a single Ni-56 mass
      decaying through :math:`^{56}\mathrm{Ni}\to{}^{56}\mathrm{Co}\to{}^{56}\mathrm{Fe}`:

      .. math::

          F_\mathrm{decay}(t) = M_\mathrm{Ni}\left[\epsilon_\mathrm{Ni}\,e^{-t/\tau_\mathrm{Ni}}
          + \epsilon_\mathrm{Co}\left(e^{-t/\tau_\mathrm{Co}} - e^{-t/\tau_\mathrm{Ni}}\right)\right],
          \qquad
          T(t) = \max\left\{\left[\frac{L(t)}{4\pi\sigma_\mathrm{SB}(v_\mathrm{ej}t)^2}\right]^{1/4},
          T_\mathrm{floor}\right\}.

      The priors on ``M_Ni``, ``M_ej`` and ``v_ej`` are the SN Ib subsample statistics (mean, sample
      standard deviation) of the analytical-model fits in Table 6 of :footcite:t:`lyman2016` (13
      events). ``kappa`` is fixed at :math:`0.06\,\mathrm{cm^2\,g^{-1}}`, the single grey optical
      opacity value that paper assumes (rather than fits) for every event in its sample.
      ``kappa_gamma`` is fixed at :math:`0.04\,\mathrm{cm^2\,g^{-1}}`, comparable to `Type Ia`'s
      Scalzo+14-derived value, since Lyman et al. 2016's own analytical model has no gamma-ray
      leakage term at all (it is the original Arnett 1982 diffusion formalism) and so gives no
      direct constraint on it. ``T_floor`` is left close to `Type Ia`'s value.

      .. dropdown:: Parameter priors

         .. list-table::
            :header-rows: 1
            :widths: 16 14 38 32

            * - Parameter
              - Symbol
              - Prior
              - Notes / Source
            * - ``M_Ni``
              - :math:`M_\mathrm{Ni}`
              - TruncatedNormal(0.17, :math:`\sigma`\=0.16; bounds :math:`[0.01, 3.0]\,M_\odot`)
              - :footcite:t:`lyman2016`, SN Ib subsample (13 events).
            * - ``M_ej``
              - :math:`M_\mathrm{ej}`
              - TruncatedNormal(2.6, :math:`\sigma`\=1.1; bounds :math:`[0.1, 8.0]\,M_\odot`)
              - :footcite:t:`lyman2016`, SN Ib subsample.
            * - ``v_ej``
              - :math:`v_\mathrm{ej}`
              - TruncatedNormal(9.9, :math:`\sigma`\=1.4; bounds :math:`[4, 16]\times10^3\ \mathrm{km\,s^{-1}}`)
              - :footcite:t:`lyman2016`, SN Ib subsample photospheric velocities.
            * - ``kappa``
              - :math:`\kappa`
              - Fixed, 0.06 :math:`\mathrm{cm^2\,g^{-1}}`
              - :footcite:t:`lyman2016`'s assumed (not fit) value.
            * - ``kappa_gamma``
              - :math:`\kappa_\gamma`
              - Fixed, 0.04 :math:`\mathrm{cm^2\,g^{-1}}`
              - Not constrained by :footcite:t:`lyman2016` (no leakage term in their model);
                comparable to `Type Ia`'s :footcite:t:`scalzo2014` value.
            * - ``T_floor``
              - :math:`T_\mathrm{floor}`
              - TruncatedNormal(5000 K, :math:`\sigma`\=1000 K; bounds :math:`[3000, 10000]` K)
              - Same floor family as the other Arnett-based models on this site; not calibrated
                against Ib data specifically.

      .. rubric:: Simulated Light Curves

      The plot below draws 300 random parameter realizations from the priors above. The top panel shows
      the bolometric light curves, each shifted so that its own peak sits at zero, overlaid with the
      individual bolometric light curves of the Type Ib SNe in :footcite:t:`lyman2016`, which are measured
      relative to maximum light; this checks the *shape* of the light curve (rise, peak width and
      decline) predicted by the Arnett diffusion model against the same sample its priors are drawn
      from. The bottom panel shows the floored-photosphere blackbody temperature against time since
      explosion, overlaid with the temperatures of the Type Ib SNe in :footcite:t:`prentice2019`,
      shifted to time since explosion using the tabulated :math:`t_p`.

      .. plot::
         :include-source: false

         import numpy as np
         import matplotlib.pyplot as plt
         from astropy import units as u

         from uvex_transients.models.supernovae import TypeIbSED as SEDClass
         from uvex_transients.utils.lightcurve_archive import LightcurveArchive

         rng = np.random.default_rng(20260918)
         n_samples = 300

         params = SEDClass().sample_parameters(size=n_samples, rng=rng)
         params_grid = {name: value[:, None] for name, value in params.items()}
         archive = LightcurveArchive()

         fig, (ax_P, ax_T) = plt.subplots(2, 1, figsize=(6.4, 7.2))

         # Top panel: simulated light curves aligned on their own peak, against Lyman+16
         # (whose times are already relative to maximum light).
         t_lin = np.linspace(0.05, 100, 2000) * u.day
         L_lin = SEDClass.eval_bolometric(t_lin, **params_grid).to_value(u.erg / u.s)
         t_peak = t_lin.to_value(u.day)[np.argmax(L_lin, axis=1)]
         for row in range(n_samples):
             ax_P.plot(t_lin.to_value(u.day) - t_peak[row], L_lin[row], color="C0", lw=0.4, alpha=0.15)

         lyman_events = [name for name in archive.events("supernovae/Ib") if name.endswith("lyman2016")]
         for i, name in enumerate(lyman_events):
             lbol_obs = archive.table("supernovae/Ib", name, "L_bol")
             ax_P.plot(
                 lbol_obs["time"].to_value(u.day), lbol_obs["L_bol"].to_value(u.erg / u.s),
                 color="k", lw=0.8, marker="o", ms=2.5, alpha=0.7,
                 label="Lyman+16 (n=%d)" % len(lyman_events) if i == 0 else None,
             )
         ax_P.set_yscale("log")
         ax_P.set_xlim([-25, 70])
         ax_P.set_ylim([1e40, 10**43.5])
         ax_P.set_xlabel("Time since peak [days]")
         ax_P.set_ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
         ax_P.set_title("Type Ib: simulated bolometric light curves (n=300)")
         ax_P.legend(loc="lower right", fontsize=8, frameon=False)

         # Bottom panel: photospheric temperature against time since explosion.
         t = np.geomspace(0.5, 100, 400) * u.day
         T = SEDClass.temperature(t, **params_grid)
         for row in range(n_samples):
             ax_T.plot(t.to_value(u.day), T[row].to_value(u.K), color="C3", lw=0.4, alpha=0.15)

         # (archive key, label, marker, colour) from Prentice+19.
         observed_sne = [
             ("2015ah_prentice2019", "SN 2015ah", "o", "k"),
             ("2015ap_prentice2019", "SN 2015ap", "s", "firebrick"),
             ("2016frp_prentice2019", "SN 2016frp", "^", "C2"),
             ("2016jdw_prentice2019", "SN 2016jdw", "X", "C7"),
         ]
         for suffix, label, marker, color in observed_sne:
             Tphot_obs = archive.table("supernovae/Ib", suffix, "T_phot")
             ax_T.scatter(
                 Tphot_obs["time"].to_value(u.day), Tphot_obs["T_phot"].to_value(u.K),
                 marker=marker, s=28, color=color, edgecolor="white", linewidth=0.5, zorder=5,
                 label=label,
             )
         ax_T.set_xscale("log")
         ax_T.set_yscale("log")
         ax_T.set_xlabel("Time since explosion [days]")
         ax_T.set_ylabel("Photospheric temperature [K]")
         ax_T.set_ylim([3e3, 5e4])
         ax_T.legend(loc="upper right", fontsize=7, frameon=False)

         fig.tight_layout()

      .. rubric:: Observability Summary

      Below are the redshifts :math:`z` and corresponding bandpass peak apparent AB magnitudes
      :math:`m_\mathrm{AB}` of 1000 simulated events drawn from the priors above, with the UVEX 1
      Dwell limit of :math:`m<24.5` overplotted.

      .. plot::
         :include-source: false

         import numpy as np
         import matplotlib.pyplot as plt
         from astropy import units as u

         from m4opt.missions import uvex
         from uvex_transients.transients.supernovae import TypeIbSNe

         rng = np.random.default_rng(20260918)
         n_samples = 1000

         sn = TypeIbSNe()
         z = sn.sample_event_redshift(n_samples, rng=rng)
         params = sn.sed.sample_parameters(size=n_samples, rng=rng)
         params_grid = {name: value[:, None] for name, value in params.items()}

         t_grid_rest = np.geomspace(0.1, 100, 300) * u.day
         t_obs_grid = t_grid_rest[None, :] * (1.0 + z)[:, None]
         z_grid_bcast = np.broadcast_to(z[:, None], t_obs_grid.shape)

         bandpasses = uvex.detector.bandpasses
         band_names = list(bandpasses)

         fig, axes = plt.subplots(1, len(band_names), figsize=(10.5, 4.8), sharey=True)

         for ax, band_name in zip(axes, band_names):
             mag_curve = sn.sed.mag_bandpass(
                 bandpasses[band_name], t_obs_grid, redshift=z_grid_bcast, **params_grid
             ).to_value(u.ABmag)
             mag = np.nanmin(mag_curve, axis=1)
             finite = np.isfinite(mag)

             ax.scatter(z[finite], mag[finite], s=5, ec="k", fc="k", alpha=0.5, label="Simulated events")
             ax.axhline(24.5, color="firebrick", ls="--", lw=1.2, label="UVEX limit (1 Dwell)")

             ax.invert_yaxis()
             ax.set_xlabel("Redshift")
             ax.set_title(f"UVEX {band_name}")
             ax.legend(loc="upper right", fontsize=8, frameon=False)
             ax.set_ylim([35, 15])

         axes[0].set_ylabel("Peak apparent AB magnitude")
         fig.suptitle(f"Type Ib: peak apparent magnitude vs. redshift (n={n_samples})")
         fig.tight_layout()

      The anticipated rate detectable by UVEX at this limit is as follows, assuming that any event above
      the :math:`m<24.5` limit is detectable, and that the population is isotropic and homogeneous in
      comoving volume out to its redshift limit:

      .. plot::
         :include-source: false

         import numpy as np
         import matplotlib.pyplot as plt
         from astropy import units as u

         from m4opt.missions import uvex
         from uvex_transients.transients.supernovae import TypeIbSNe
         from uvex_transients.utils.plotting import add_funnel_legend, get_band_color, plot_rate_bars

         rng = np.random.default_rng(20260918)
         n_samples = 1000

         sn = TypeIbSNe()
         redshift = sn.sample_event_redshift(n_samples, rng=rng)
         params = sn.sed.sample_parameters(size=n_samples, rng=rng)
         params_grid = {pname: value[:, None] for pname, value in params.items()}

         # The all-sky rate, with no survey footprint applied.
         all_sky_rate = sn.all_sky_rate

         t_grid_rest = np.geomspace(0.1, 100, 300) * u.day
         t_obs_grid = t_grid_rest[None, :] * (1.0 + redshift)[:, None]
         z_grid_bcast = np.broadcast_to(redshift[:, None], t_obs_grid.shape)

         visible_counts = {}
         for band_name, bandpass in uvex.detector.bandpasses.items():
             mag_curve = sn.sed.mag_bandpass(
                 bandpass,
                 t_obs_grid,
                 redshift=z_grid_bcast,
                 **params_grid,
             ).to_value(u.ABmag)
             magnitudes = np.nanmin(mag_curve, axis=1)

             visible = magnitudes < 24.5
             visible_counts[band_name] = int(np.count_nonzero(visible))

             print(
                 f"{band_name}: {visible_counts[band_name] / n_samples * all_sky_rate:.2f} "
                 f"({visible_counts[band_name] / n_samples:.1%} of events visible)"
             )

         # Plot all-sky visible rates, with MC (statistical) and rate (systematic)
         # uncertainty -- see uvex_transients.utils.plotting.plot_rate_bars.
         band_names = list(visible_counts)

         fig, ax = plt.subplots(figsize=(5, 4))
         plot_rate_bars(
             ax,
             band_names,
             [visible_counts[band] for band in band_names],
             n_samples,
             all_sky_rate,
             rate_ci=sn.RATE_CI,
             color=[get_band_color(band) for band in band_names],
         )
         ax.set_yscale("log")
         ax.set_ylabel(r"All-sky rate [yr$^{-1}$]")
         ax.set_title("Peak-visible Type Ib rate")
         add_funnel_legend(ax, loc="lower right")

         fig.tight_layout()

   .. tab-item:: Type Ic

      Implemented by :class:`~uvex_transients.transients.supernovae.TypeIcSNe`, pairing
      :class:`~uvex_transients.models.supernovae.Ibc.TypeIcSED` with the rate/duration metadata
      described below.

      .. rubric:: Quick Facts

      .. list-table::
         :header-rows: 1
         :widths: 15 25 15 45

         * - Quantity
           - Value
           - Source
           - Notes
         * - Rate
           - :math:`R_\mathrm{CC}(z) = k\,\psi_\mathrm{UV}(z)`; Type Ic 6.5% of :math:`R_\mathrm{CC}(z)`
           - :footcite:t:`strolger2015`, :footcite:t:`madau2014`,
             :footcite:t:`shivvers2017`
           - Tracks the cosmic star-formation history. :footcite:t:`shivvers2017` find Ic is
             :math:`21.5\pm8.6\%` of the stripped-envelope (SESNe) rate, which is itself
             :math:`30.4^{+5.0}_{-4.9}\%` of the total core-collapse rate, so the Type Ic fraction
             is :math:`0.215\times0.304=0.065`. Combined in quadrature with
             :footcite:t:`strolger2015`'s :math:`+27\%/-31\%` normalization uncertainty, this gives
             :attr:`~uvex_transients.transients.supernovae.TypeIcSNe.RATE_CI` (see
             :ref:`user_guide_transients_rate_uncertainty`).
         * - Redshift limit
           - :math:`z = 0.5`
           - --
           - Below the Type IIP and IIb limits, since these events peak at lower luminosity and are cool
             in the UV: in the observability check below, no simulated event beyond
             :math:`z \approx 0.5` clears the UVEX limit.
         * - Duration
           - 100 days
           - --
           - Covers the rise, peak (:math:`t_p \approx 17` d after explosion, prior median) and the decline.

      .. rubric:: SED Model

      *Model Class*: :class:`~uvex_transients.models.supernovae.Ibc.TypeIcSED`

      :class:`~uvex_transients.models.supernovae.Ibc.TypeIcSED` shares its construction entirely
      with `Type Ib`'s :class:`~uvex_transients.models.supernovae.Ibc.TypeIbSED` (see its SED Model
      section above for the :math:`F_\mathrm{decay}(t)`/:math:`T(t)` math and the reasoning behind
      ``kappa``, ``kappa_gamma`` and ``T_floor``), differing only in that ``M_Ni``, ``M_ej`` and
      ``v_ej`` are drawn from the SN Ic subsample statistics of Table 6 of :footcite:t:`lyman2016`
      (8 events) instead of the Type Ib ones.

      .. dropdown:: Parameter priors

         .. list-table::
            :header-rows: 1
            :widths: 16 14 38 32

            * - Parameter
              - Symbol
              - Prior
              - Notes / Source
            * - ``M_Ni``
              - :math:`M_\mathrm{Ni}`
              - TruncatedNormal(0.22, :math:`\sigma`\=0.16; bounds :math:`[0.01, 3.0]\,M_\odot`)
              - :footcite:t:`lyman2016`, SN Ic subsample (8 events).
            * - ``M_ej``
              - :math:`M_\mathrm{ej}`
              - TruncatedNormal(3.0, :math:`\sigma`\=2.8; bounds :math:`[0.1, 6.0]\,M_\odot`)
              - :footcite:t:`lyman2016`, SN Ic subsample.
            * - ``v_ej``
              - :math:`v_\mathrm{ej}`
              - TruncatedNormal(10.4, :math:`\sigma`\=1.2; bounds :math:`[4, 16]\times10^3\ \mathrm{km\,s^{-1}}`)
              - :footcite:t:`lyman2016`, SN Ic subsample photospheric velocities.
            * - ``kappa``
              - :math:`\kappa`
              - Fixed, 0.06 :math:`\mathrm{cm^2\,g^{-1}}`
              - :footcite:t:`lyman2016`'s assumed (not fit) value.
            * - ``kappa_gamma``
              - :math:`\kappa_\gamma`
              - Fixed, 0.04 :math:`\mathrm{cm^2\,g^{-1}}`
              - Not constrained by :footcite:t:`lyman2016` (no leakage term in their model);
                comparable to `Type Ia`'s :footcite:t:`scalzo2014` value.
            * - ``T_floor``
              - :math:`T_\mathrm{floor}`
              - TruncatedNormal(6000 K, :math:`\sigma`\=1000 K; bounds :math:`[3000, 10000]` K)
              - Same floor family as the other Arnett-based models on this site; not calibrated
                against Ic data specifically.

      .. rubric:: Simulated Light Curves

      The plot below draws 300 random parameter realizations from the priors above. The top panel shows
      the bolometric light curves, each shifted so that its own peak sits at zero, overlaid with the
      individual bolometric light curves of the Type Ic SNe in :footcite:t:`lyman2016`, which are measured
      relative to maximum light; this checks the *shape* of the light curve (rise, peak width and
      decline) predicted by the Arnett diffusion model against the same sample its priors are drawn
      from. The bottom panel shows the floored-photosphere blackbody temperature against time since
      explosion, overlaid with the temperatures of the Type Ic SNe in :footcite:t:`prentice2019`,
      shifted to time since explosion using the tabulated :math:`t_p`.

      .. plot::
         :include-source: false

         import numpy as np
         import matplotlib.pyplot as plt
         from astropy import units as u

         from uvex_transients.models.supernovae import TypeIcSED as SEDClass
         from uvex_transients.utils.lightcurve_archive import LightcurveArchive

         rng = np.random.default_rng(20260918)
         n_samples = 300

         params = SEDClass().sample_parameters(size=n_samples, rng=rng)
         params_grid = {name: value[:, None] for name, value in params.items()}
         archive = LightcurveArchive()

         fig, (ax_P, ax_T) = plt.subplots(2, 1, figsize=(6.4, 7.2))

         # Top panel: simulated light curves aligned on their own peak, against Lyman+16
         # (whose times are already relative to maximum light).
         t_lin = np.linspace(0.05, 100, 2000) * u.day
         L_lin = SEDClass.eval_bolometric(t_lin, **params_grid).to_value(u.erg / u.s)
         t_peak = t_lin.to_value(u.day)[np.argmax(L_lin, axis=1)]
         for row in range(n_samples):
             ax_P.plot(t_lin.to_value(u.day) - t_peak[row], L_lin[row], color="C0", lw=0.4, alpha=0.15)

         lyman_events = [name for name in archive.events("supernovae/Ic") if name.endswith("lyman2016")]
         for i, name in enumerate(lyman_events):
             lbol_obs = archive.table("supernovae/Ic", name, "L_bol")
             ax_P.plot(
                 lbol_obs["time"].to_value(u.day), lbol_obs["L_bol"].to_value(u.erg / u.s),
                 color="k", lw=0.8, marker="o", ms=2.5, alpha=0.7,
                 label="Lyman+16 (n=%d)" % len(lyman_events) if i == 0 else None,
             )
         ax_P.set_yscale("log")
         ax_P.set_xlim([-25, 70])
         ax_P.set_ylim([1e40, 10**43.5])
         ax_P.set_xlabel("Time since peak [days]")
         ax_P.set_ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
         ax_P.set_title("Type Ic: simulated bolometric light curves (n=300)")
         ax_P.legend(loc="lower right", fontsize=8, frameon=False)

         # Bottom panel: photospheric temperature against time since explosion.
         t = np.geomspace(0.5, 100, 400) * u.day
         T = SEDClass.temperature(t, **params_grid)
         for row in range(n_samples):
             ax_T.plot(t.to_value(u.day), T[row].to_value(u.K), color="C3", lw=0.4, alpha=0.15)

         # (archive key, label, marker, colour) from Prentice+19.
         observed_sne = [
             ("2013F_prentice2019", "SN 2013F", "o", "k"),
             ("2016P_prentice2019", "SN 2016P", "s", "firebrick"),
             ("2016iae_prentice2019", "SN 2016iae", "^", "C2"),
             ("2017dcc_prentice2019", "SN 2017dcc", "X", "C7"),
             ("2017ifh_prentice2019", "SN 2017ifh", "*", "C8"),
             ("2018ie_prentice2019", "SN 2018ie", "D", "C4"),
         ]
         for suffix, label, marker, color in observed_sne:
             Tphot_obs = archive.table("supernovae/Ic", suffix, "T_phot")
             ax_T.scatter(
                 Tphot_obs["time"].to_value(u.day), Tphot_obs["T_phot"].to_value(u.K),
                 marker=marker, s=28, color=color, edgecolor="white", linewidth=0.5, zorder=5,
                 label=label,
             )
         ax_T.set_xscale("log")
         ax_T.set_yscale("log")
         ax_T.set_xlabel("Time since explosion [days]")
         ax_T.set_ylabel("Photospheric temperature [K]")
         ax_T.set_ylim([3e3, 5e4])
         ax_T.legend(loc="upper right", fontsize=7, frameon=False)

         fig.tight_layout()

      .. rubric:: Observability Summary

      Below are the redshifts :math:`z` and corresponding bandpass peak apparent AB magnitudes
      :math:`m_\mathrm{AB}` of 1000 simulated events drawn from the priors above, with the UVEX 1
      Dwell limit of :math:`m<24.5` overplotted.

      .. plot::
         :include-source: false

         import numpy as np
         import matplotlib.pyplot as plt
         from astropy import units as u

         from m4opt.missions import uvex
         from uvex_transients.transients.supernovae import TypeIcSNe

         rng = np.random.default_rng(20260918)
         n_samples = 1000

         sn = TypeIcSNe()
         z = sn.sample_event_redshift(n_samples, rng=rng)
         params = sn.sed.sample_parameters(size=n_samples, rng=rng)
         params_grid = {name: value[:, None] for name, value in params.items()}

         t_grid_rest = np.geomspace(0.1, 100, 300) * u.day
         t_obs_grid = t_grid_rest[None, :] * (1.0 + z)[:, None]
         z_grid_bcast = np.broadcast_to(z[:, None], t_obs_grid.shape)

         bandpasses = uvex.detector.bandpasses
         band_names = list(bandpasses)

         fig, axes = plt.subplots(1, len(band_names), figsize=(10.5, 4.8), sharey=True)

         for ax, band_name in zip(axes, band_names):
             mag_curve = sn.sed.mag_bandpass(
                 bandpasses[band_name], t_obs_grid, redshift=z_grid_bcast, **params_grid
             ).to_value(u.ABmag)
             mag = np.nanmin(mag_curve, axis=1)
             finite = np.isfinite(mag)

             ax.scatter(z[finite], mag[finite], s=5, ec="k", fc="k", alpha=0.5, label="Simulated events")
             ax.axhline(24.5, color="firebrick", ls="--", lw=1.2, label="UVEX limit (1 Dwell)")

             ax.invert_yaxis()
             ax.set_xlabel("Redshift")
             ax.set_title(f"UVEX {band_name}")
             ax.legend(loc="upper right", fontsize=8, frameon=False)
             ax.set_ylim([35, 15])

         axes[0].set_ylabel("Peak apparent AB magnitude")
         fig.suptitle(f"Type Ic: peak apparent magnitude vs. redshift (n={n_samples})")
         fig.tight_layout()

      The anticipated rate detectable by UVEX at this limit is as follows, assuming that any event above
      the :math:`m<24.5` limit is detectable, and that the population is isotropic and homogeneous in
      comoving volume out to its redshift limit:

      .. plot::
         :include-source: false

         import numpy as np
         import matplotlib.pyplot as plt
         from astropy import units as u

         from m4opt.missions import uvex
         from uvex_transients.transients.supernovae import TypeIcSNe
         from uvex_transients.utils.plotting import add_funnel_legend, get_band_color, plot_rate_bars

         rng = np.random.default_rng(20260918)
         n_samples = 1000

         sn = TypeIcSNe()
         redshift = sn.sample_event_redshift(n_samples, rng=rng)
         params = sn.sed.sample_parameters(size=n_samples, rng=rng)
         params_grid = {pname: value[:, None] for pname, value in params.items()}

         # The all-sky rate, with no survey footprint applied.
         all_sky_rate = sn.all_sky_rate

         t_grid_rest = np.geomspace(0.1, 100, 300) * u.day
         t_obs_grid = t_grid_rest[None, :] * (1.0 + redshift)[:, None]
         z_grid_bcast = np.broadcast_to(redshift[:, None], t_obs_grid.shape)

         visible_counts = {}
         for band_name, bandpass in uvex.detector.bandpasses.items():
             mag_curve = sn.sed.mag_bandpass(
                 bandpass,
                 t_obs_grid,
                 redshift=z_grid_bcast,
                 **params_grid,
             ).to_value(u.ABmag)
             magnitudes = np.nanmin(mag_curve, axis=1)

             visible = magnitudes < 24.5
             visible_counts[band_name] = int(np.count_nonzero(visible))

             print(
                 f"{band_name}: {visible_counts[band_name] / n_samples * all_sky_rate:.2f} "
                 f"({visible_counts[band_name] / n_samples:.1%} of events visible)"
             )

         # Plot all-sky visible rates, with MC (statistical) and rate (systematic)
         # uncertainty -- see uvex_transients.utils.plotting.plot_rate_bars.
         band_names = list(visible_counts)

         fig, ax = plt.subplots(figsize=(5, 4))
         plot_rate_bars(
             ax,
             band_names,
             [visible_counts[band] for band in band_names],
             n_samples,
             all_sky_rate,
             rate_ci=sn.RATE_CI,
             color=[get_band_color(band) for band in band_names],
         )
         ax.set_yscale("log")
         ax.set_ylabel(r"All-sky rate [yr$^{-1}$]")
         ax.set_title("Peak-visible Type Ic rate")
         add_funnel_legend(ax, loc="lower right")

         fig.tight_layout()

   .. tab-item:: Type Ic-BL

      Type Ic-BL (broad-lined) supernovae are Type Ic explosions with unusually high kinetic
      energies and ejecta velocities, identified spectroscopically by their broad, blended
      absorption features. Like `Type Ib`/`Type Ic` above and `Type Ia`, this population's SED
      follows the same first-principles Arnett-style radioactive-decay model, but with its own
      priors, calibrated from an explosion-property table (nickel mass, ejecta mass, photospheric
      velocity) specific to its own ZTF sample rather than the Lyman et al. 2016 sample used for
      `Type Ib`/`Type Ic`.

      Implemented by :class:`~uvex_transients.transients.supernovae.TypeIcBLSNe`, pairing
      :class:`~uvex_transients.models.supernovae.IcBL.TypeIcBLSED` with the rate/duration
      metadata described below.

      .. rubric:: Quick Facts

      .. list-table::
         :header-rows: 1
         :widths: 15 25 15 45

         * - Quantity
           - Value
           - Source
           - Notes
         * - Rate
           - :math:`R_\mathrm{CC}(z) = k\,\psi_\mathrm{UV}(z)`; Type Ic-BL 1.1% of :math:`R_\mathrm{CC}(z)`
           - :footcite:t:`strolger2015`, :footcite:t:`madau2014`,
             :footcite:t:`shivvers2017`
           - Tracks the cosmic star-formation history. :footcite:t:`shivvers2017` find Ic-BL is
             :math:`3.7^{+2.9}_{-3.7}\%` of the stripped-envelope (SESNe) rate, which is itself
             :math:`30.4^{+5.0}_{-4.9}\%` of the total core-collapse rate, so the Type Ic-BL
             fraction is :math:`0.037\times0.304=0.0112`. Combined in quadrature with
             :footcite:t:`strolger2015`'s :math:`+27\%/-31\%` normalization uncertainty, this gives
             :attr:`~uvex_transients.transients.supernovae.TypeIcBLSNe.RATE_CI` (see
             :ref:`user_guide_transients_rate_uncertainty`). The lower bound on the Ic-BL-of-SESNe
             fraction is itself consistent with zero, so the combined lower `RATE_CI` factor is
             slightly negative; treat the lower bound as effectively zero rather than literally.
         * - Redshift limit
           - :math:`z = 1`
           - --
           - Wider than the Type Ib/Type Ic limit, to cover this population's higher ejecta
             velocities and kinetic energies; in the observability check below, no simulated event
             beyond :math:`z \approx 0.55` clears the UVEX limit.
         * - Duration
           - 100 days
           - --
           - Covers the rise, peak and decline, matching `Type Ib`/`Type Ic`.

      .. rubric:: SED Model

      *Model Class*: :class:`~uvex_transients.models.supernovae.IcBL.TypeIcBLSED`

      :class:`~uvex_transients.models.supernovae.IcBL.TypeIcBLSED` reuses
      :class:`~uvex_transients.models.arnett.ArnettDecaySED`'s Arnett-style radioactive-decay
      diffusion light curve and floored-photosphere blackbody entirely (the same
      :math:`L(t)`/:math:`T(t)` machinery documented for `Type Ia` above and for the SLSNe-I model
      in :ref:`transients_slsne`):

      .. math::

          F_\mathrm{decay}(t) = M_\mathrm{Ni}\left[\epsilon_\mathrm{Ni}\,e^{-t/\tau_\mathrm{Ni}}
          + \epsilon_\mathrm{Co}\left(e^{-t/\tau_\mathrm{Co}} - e^{-t/\tau_\mathrm{Ni}}\right)\right],
          \qquad
          T(t) = \max\left\{\left[\frac{L(t)}{4\pi\sigma_\mathrm{SB}(v_\mathrm{ej}t)^2}\right]^{1/4},
          T_\mathrm{floor}\right\}.

      The priors on ``M_Ni``, ``M_ej`` and ``v_ej`` are sample statistics (mean, sample standard
      deviation) of the 36-event explosion-property table of :footcite:t:`srinivasaragavan2024`
      (nickel mass, kinetic energy, ejecta mass and photospheric velocity per event). One event
      (SN 2020wgz), whose reported
      :math:`M_\mathrm{Ni}=2.46\,M_\odot` is a >5-sigma outlier driven by an ``e_k``/``m_ej`` *lower
      limit* rather than a measurement, is excluded from the ``M_Ni`` statistics; ``M_ej`` uses only
      rows with a measured (non-lower-limit) value; ``v_ej`` is identified with the sample's
      photospheric velocities (``v_ph``), regardless of the epoch quoted. ``kappa`` and
      ``T_floor`` are not constrained by that table and are left at `Type Ia`'s values;
      ``kappa_gamma`` is instead fixed at a large value (full gamma-ray trapping across the
      simulated window), unlike `Type Ia`'s Scalzo+14 value.

      .. dropdown:: Parameter priors

         .. list-table::
            :header-rows: 1
            :widths: 16 14 38 32

            * - Parameter
              - Symbol
              - Prior
              - Notes / Source
            * - ``M_Ni``
              - :math:`M_\mathrm{Ni}`
              - TruncatedNormal(0.33, :math:`\sigma`\=0.24; bounds :math:`[0.02, 2.0]\,M_\odot`)
              - :footcite:t:`srinivasaragavan2024` (SN 2020wgz excluded, see above).
            * - ``M_ej``
              - :math:`M_\mathrm{ej}`
              - TruncatedNormal(2.54, :math:`\sigma`\=1.95; bounds :math:`[0.1, 10.0]\,M_\odot`)
              - :footcite:t:`srinivasaragavan2024`, measured (non-lower-limit) rows only.
            * - ``v_ej``
              - :math:`v_\mathrm{ej}`
              - TruncatedNormal(20.1, :math:`\sigma`\=4.96; bounds :math:`[5, 45]\times10^3\ \mathrm{km\,s^{-1}}`)
              - Sample photospheric velocities (``v_ph``).
            * - ``kappa``
              - :math:`\kappa`
              - Uniform(0.05, 0.15) :math:`\mathrm{cm^2\,g^{-1}}`
              - Not constrained by the table; same range as `Type Ia`.
            * - ``kappa_gamma``
              - :math:`\kappa_\gamma`
              - Fixed, 1000 :math:`\mathrm{cm^2\,g^{-1}}`
              - Approximates full gamma-ray trapping across the simulated window; unlike `Type Ia`,
                not :footcite:t:`scalzo2014`'s value.
            * - ``T_floor``
              - :math:`T_\mathrm{floor}`
              - TruncatedNormal(6000 K, :math:`\sigma`\=1000 K; bounds :math:`[3000, 10000]` K)
              - Same floor as the other Arnett-based models on this site; not calibrated against
                Ic-BL data specifically.

      .. rubric:: Simulated Light Curves

      The plot below draws 300 random parameter realizations from the priors above and shows the
      resulting bolometric light curves and photospheric temperatures. The top panel is overlaid
      with the individual bolometric light curves (time since explosion) of the 26 Type Ic-BL SNe
      of :footcite:t:`srinivasaragavan2024` with a full explosion-property fit (the same sample the
      priors above are derived from). No photospheric temperature data is bundled for Type Ic-BL,
      so the bottom panel is unadorned.

      .. plot::
         :include-source: false

         import numpy as np
         import matplotlib.pyplot as plt
         from astropy import units as u

         from uvex_transients.models.supernovae import TypeIcBLSED as SEDClass
         from uvex_transients.utils.lightcurve_archive import LightcurveArchive

         rng = np.random.default_rng(20260924)
         n_samples = 300

         params = SEDClass().sample_parameters(size=n_samples, rng=rng)
         params_grid = {name: value[:, None] for name, value in params.items()}
         archive = LightcurveArchive()

         t = np.geomspace(0.1, 1000, 400) * u.day
         L_bol = SEDClass.eval_bolometric(t, **params_grid)
         T = SEDClass.temperature(t, **params_grid)

         fig, (ax_L, ax_T) = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True)

         for row in range(n_samples):
             ax_L.plot(t.to_value(u.day), L_bol[row].to_value(u.erg / u.s), color="C0", lw=0.4, alpha=0.15)
             ax_T.plot(t.to_value(u.day), T[row].to_value(u.K), color="C3", lw=0.4, alpha=0.15)

         icbl_events = [name for name in archive.events("supernovae/Ic-BL") if name.endswith("srinivasaragavan2024")]
         for i, name in enumerate(icbl_events):
             lbol_obs = archive.table("supernovae/Ic-BL", name, "L_bol")
             ax_L.plot(
                 lbol_obs["time"].to_value(u.day), lbol_obs["L_bol"].to_value(u.erg / u.s),
                 color="k", lw=0.8, alpha=0.4,
                 label="Srinivasaragavan+24 (n=%d)" % len(icbl_events) if i == 0 else None,
             )

         ax_L.set_xscale("log")
         ax_L.set_yscale("log")
         ax_L.set_xlim(1e-1, 1e3)
         ax_L.set_ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
         ax_L.set_title(f"Type Ic-BL: simulated bolometric light curves (n={n_samples})")
         ax_L.legend(loc="lower left", fontsize=8, frameon=False)

         ax_T.set_xscale("log")
         ax_T.set_yscale("log")
         ax_T.set_xlim(1e-1, 1e3)
         ax_T.set_xlabel("Time since explosion [days]")
         ax_T.set_ylabel("Photospheric temperature [K]")
         ax_T.set_title(f"Type Ic-BL: simulated photospheric temperatures (n={n_samples})")

         fig.tight_layout()

      .. rubric:: Observability Summary

      Below are the redshifts :math:`z` and corresponding bandpass peak apparent AB magnitudes
      :math:`m_\mathrm{AB}` of 1000 simulated events drawn from the priors above, with the UVEX 1
      Dwell limit of :math:`m<24.5` overplotted.

      .. plot::
         :include-source: false

         import numpy as np
         import matplotlib.pyplot as plt
         from astropy import units as u

         from m4opt.missions import uvex
         from uvex_transients.transients.supernovae import TypeIcBLSNe

         rng = np.random.default_rng(20260924)
         n_samples = 1000

         sn = TypeIcBLSNe()
         z = sn.sample_event_redshift(n_samples, rng=rng)
         params = sn.sed.sample_parameters(size=n_samples, rng=rng)
         params_grid = {name: value[:, None] for name, value in params.items()}

         t_grid_rest = np.geomspace(0.1, 100, 300) * u.day
         t_obs_grid = t_grid_rest[None, :] * (1.0 + z)[:, None]
         z_grid_bcast = np.broadcast_to(z[:, None], t_obs_grid.shape)

         bandpasses = uvex.detector.bandpasses
         band_names = list(bandpasses)

         fig, axes = plt.subplots(1, len(band_names), figsize=(10.5, 4.8), sharey=True)

         for ax, band_name in zip(axes, band_names):
             mag_curve = sn.sed.mag_bandpass(
                 bandpasses[band_name], t_obs_grid, redshift=z_grid_bcast, **params_grid
             ).to_value(u.ABmag)
             mag = np.nanmin(mag_curve, axis=1)
             finite = np.isfinite(mag)

             ax.scatter(z[finite], mag[finite], s=5, ec="k", fc="k", alpha=0.5, label="Simulated events")
             ax.axhline(24.5, color="firebrick", ls="--", lw=1.2, label="UVEX limit (1 Dwell)")

             ax.invert_yaxis()
             ax.set_xlabel("Redshift")
             ax.set_title(f"UVEX {band_name}")
             ax.legend(loc="upper right", fontsize=8, frameon=False)
             ax.set_ylim([35, 15])

         axes[0].set_ylabel("Peak apparent AB magnitude")
         fig.suptitle(f"Type Ic-BL: peak apparent magnitude vs. redshift (n={n_samples})")
         fig.tight_layout()

      The anticipated rate detectable by UVEX at this limit is as follows, assuming that any event above
      the :math:`m<24.5` limit is detectable, and that the population is isotropic and homogeneous in
      comoving volume out to its redshift limit:

      .. plot::
         :include-source: false

         import numpy as np
         import matplotlib.pyplot as plt
         from astropy import units as u

         from m4opt.missions import uvex
         from uvex_transients.transients.supernovae import TypeIcBLSNe
         from uvex_transients.utils.plotting import add_funnel_legend, get_band_color, plot_rate_bars

         rng = np.random.default_rng(20260924)
         n_samples = 1000

         sn = TypeIcBLSNe()
         redshift = sn.sample_event_redshift(n_samples, rng=rng)
         params = sn.sed.sample_parameters(size=n_samples, rng=rng)
         params_grid = {pname: value[:, None] for pname, value in params.items()}

         # The all-sky rate, with no survey footprint applied.
         all_sky_rate = sn.all_sky_rate

         t_grid_rest = np.geomspace(0.1, 100, 300) * u.day
         t_obs_grid = t_grid_rest[None, :] * (1.0 + redshift)[:, None]
         z_grid_bcast = np.broadcast_to(redshift[:, None], t_obs_grid.shape)

         visible_counts = {}
         for band_name, bandpass in uvex.detector.bandpasses.items():
             mag_curve = sn.sed.mag_bandpass(
                 bandpass,
                 t_obs_grid,
                 redshift=z_grid_bcast,
                 **params_grid,
             ).to_value(u.ABmag)
             magnitudes = np.nanmin(mag_curve, axis=1)

             visible = magnitudes < 24.5
             visible_counts[band_name] = int(np.count_nonzero(visible))

             print(
                 f"{band_name}: {visible_counts[band_name] / n_samples * all_sky_rate:.2f} "
                 f"({visible_counts[band_name] / n_samples:.1%} of events visible)"
             )

         # Plot all-sky visible rates, with MC (statistical) and rate (systematic)
         # uncertainty -- see uvex_transients.utils.plotting.plot_rate_bars.
         band_names = list(visible_counts)

         fig, ax = plt.subplots(figsize=(5, 4))
         plot_rate_bars(
             ax,
             band_names,
             [visible_counts[band] for band in band_names],
             n_samples,
             all_sky_rate,
             rate_ci=sn.RATE_CI,
             color=[get_band_color(band) for band in band_names],
         )
         ax.set_yscale("log")
         ax.set_ylabel(r"All-sky rate [yr$^{-1}$]")
         ax.set_title("Peak-visible Type Ic-BL rate")
         add_funnel_legend(ax, loc="lower right")

         fig.tight_layout()

References
-----------

.. footbibliography::
