.. _transients_type_i:

Type I (Stripped-Envelope) Supernovae
=======================================

Type Ib and Type Ic supernovae are core-collapse explosions of massive stars that lost their
hydrogen envelope (Ib: helium remains; Ic: helium is also stripped) before exploding. Their light
curves are powered by the radioactive decay of :math:`^{56}\mathrm{Ni}` and are typically a single
peak, roughly two to three weeks after explosion, followed by a decline; unlike Type IIb SNe
(:ref:`transients_type_ii`) they lack a distinct early shock-cooling peak. Both are modeled here by
the same phenomenological form -- a single Bazin pulse times a cooling blackbody photosphere -- and
differ only in their event rates, implemented as two sibling transient populations below.

.. note::

   The priors are phenomenological and calibrated against two samples: the full bolometric light
   curves of :footcite:t:`lyman2016` (13 Ib and 8 Ic events, aligned on the epoch of maximum rather
   than on explosion), and the photospheric temperature curves of the Type Ib and Ic SNe of
   :footcite:t:`prentice2019` (four Ib and six Ic events, placed on a time-since-explosion axis using
   their tabulated times of peak). The samples are small, so the ranges are deliberately broad. Ib and
   Ic currently share the same priors and differ only in their event rates.

.. tab-set::

   .. tab-item:: Type Ib

      Implemented by :class:`~uvex_transients.transients.supernovae.TypeIbSNe`, pairing
      :class:`~uvex_transients.models.supernovae.Ibc.TypeIbSED` with the rate/duration metadata
      described below.

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
           - :math:`R_\mathrm{CC}(z) = k h^2 \psi_\mathrm{UV}(z)`; Type Ib 4.9% of :math:`R_\mathrm{CC}(z)`
           - :footcite:t:`strolger2015`, :footcite:t:`madau2014`,
             :footcite:t:`shivvers2017`
           - Tracks the cosmic star-formation history. Stripped-envelope SNe are 30.4% of core-collapse
             SNe :footcite:p:`shivvers2017`, of which 16.1% are Type Ib :footcite:p:`shivvers2017`, so the Type Ib fraction is
             :math:`0.304 \times 0.161 = 0.049`.
         * - Redshift limit
           - :math:`z = 0.5`
           - --
           - Below the Type IIP and IIb limits, since these events peak at lower luminosity and are cool
             in the UV: in the observability check below, no simulated event beyond
             :math:`z \approx 0.4` clears the UVEX limit.
         * - Duration
           - 100 days
           - --
           - Covers the rise, peak (:math:`t_p \approx 20` d after explosion) and the decline.

      SED Model
      ----------

      :class:`~uvex_transients.models.supernovae.Ibc.TypeIbSED` is a single Bazin pulse times the
      same single-power-law cooling blackbody photosphere used elsewhere in this package (e.g.
      :class:`~uvex_transients.models.supernovae.VillarCoolingBlackbodySED`):

      .. math::

          L_\mathrm{bol}(t) =
          A\,
          \frac{\exp[-(t-t_0)/\tau_\mathrm{fall}]}{1 + \exp[-(t-t_0)/\tau_\mathrm{rise}]},
          \qquad
          T(t) = T_\mathrm{floor} + (T_0 - T_\mathrm{floor})\left(1 + \frac{t}{\tau_T}\right)^{-\alpha_T}.

      The light curve is delegated to
      :class:`~uvex_transients.models.lightcurves.generic.BazinLightcurve`, with the logistic rise
      timescale tied to the transition time, :math:`\tau_\mathrm{rise} = t_0/2.5`. This removes one free
      parameter and, since the rise shape then scales with :math:`t_0`, lets :math:`t_0` alone set the
      time to peak,

      .. math::

          t_p = t_0\left[1 + 0.4\ln\!\left(\frac{2.5\,\tau_\mathrm{fall}}{t_0} - 1\right)\right]
          \approx 1.9\,t_0,

      and leaves the luminosity at :math:`t = 0` at about 14% of the peak. The amplitude :math:`A` is
      a normalization, not the peak luminosity. The priors on :math:`t_0` and :math:`\tau_\mathrm{fall}`
      give a time to peak of about 20 days (5th to 95th percentile 17.8--21.9 d).

      .. dropdown:: Parameter priors

         ``alpha_T`` is held fixed. Fits of the cooling law to the observed temperatures prefer a
         fast, close-to-exponential relaxation to a floor, i.e. a large :math:`\alpha_T` with a long
         :math:`\tau_T`, which are strongly degenerate with the temperature data starting 5--15 days
         after explosion, so one of the two is fixed and the other carries the scatter.

         .. list-table::
            :header-rows: 1
            :widths: 16 12 26 46

            * - Parameter
              - Symbol
              - Prior
              - Notes
            * - ``amplitude``
              - :math:`A`
              - Normal(:math:`\log_{10}(A/\mathrm{erg\,s^{-1}})`; mean=42.6, :math:`\sigma`\=0.3)
              - Bazin normalization; induces a peak :math:`\log_{10} L_p \approx 42.45`.
            * - ``t0``
              - :math:`t_0`
              - Uniform(9.5 d, 12 d)
              - Transition time of the pulse; also sets the rise timescale, :math:`\tau_\mathrm{rise} = t_0/2.5`.
            * - ``fall``
              - :math:`\tau_\mathrm{fall}`
              - Uniform(30 d, 50 d)
              - Exponential decline timescale.
            * - ``T0``
              - :math:`T_0`
              - Normal(:math:`\log_{10}(T_0/\mathrm{K})`; mean=4.2, :math:`\sigma`\=0.1)
              - Photospheric temperature as :math:`t \to 0`.
            * - ``T_floor``
              - :math:`T_\mathrm{floor}`
              - Uniform(4000 K, 5200 K)
              - Asymptotic late-time photospheric temperature.
            * - ``tau_T``
              - :math:`\tau_T`
              - Uniform(15 d, 45 d)
              - Photospheric cooling timescale.
            * - ``alpha_T``
              - :math:`\alpha_T`
              - Fixed (4)
              - Photospheric cooling power-law index.

      Simulated Light Curves
      ~~~~~~~~~~~~~~~~~~~~~~~

      The plot below draws 300 random parameter realizations from the priors above. The top panel shows
      the bolometric light curves, each shifted so that its own peak sits at zero, overlaid with the
      individual bolometric light curves of the Type Ib SNe in :footcite:t:`lyman2016`, which are measured
      relative to maximum light; this checks the *shape* of the light curve (rise, peak width and
      decline). The bottom panel shows the photospheric temperature against time since explosion,
      overlaid with the temperatures of the Type Ib SNe in :footcite:t:`prentice2019`, shifted to time
      since explosion using the tabulated :math:`t_p`.

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

      Observability Summary
      ----------------------

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

         rng = np.random.default_rng(20260918)
         n_samples = 1000

         sn = TypeIbSNe()
         redshift = sn.sample_event_redshift(n_samples, rng=rng)
         params = sn.sed.sample_parameters(size=n_samples, rng=rng)
         params_grid = {pname: value[:, None] for pname, value in params.items()}

         # Convert the integrated rate per steradian to an all-sky rate.
         all_sky_rate = 4 * np.pi * sn.integrated_event_rate * u.sr

         t_grid_rest = np.geomspace(0.1, 100, 300) * u.day
         t_obs_grid = t_grid_rest[None, :] * (1.0 + redshift)[:, None]
         z_grid_bcast = np.broadcast_to(redshift[:, None], t_obs_grid.shape)

         visible_rates = {}
         for band_name, bandpass in uvex.detector.bandpasses.items():
             mag_curve = sn.sed.mag_bandpass(
                 bandpass,
                 t_obs_grid,
                 redshift=z_grid_bcast,
                 **params_grid,
             ).to_value(u.ABmag)
             magnitudes = np.nanmin(mag_curve, axis=1)

             visible = magnitudes < 24.5
             visible_fraction = np.mean(visible)
             visible_rate = visible_fraction * all_sky_rate

             visible_rates[band_name] = visible_rate.to_value(1 / u.yr)

             print(
                 f"{band_name}: {visible_rate:.2f} "
                 f"({visible_fraction:.1%} of events visible)"
             )

         fig, ax = plt.subplots(figsize=(5, 4))
         ax.bar(list(visible_rates), list(visible_rates.values()), color=["C0", "C1"])
         ax.set_yscale("log")
         ax.set_ylabel(r"All-sky rate [yr$^{-1}$]")
         ax.set_title("Peak-visible Type Ib rate")

         fig.tight_layout()

   .. tab-item:: Type Ic

      Implemented by :class:`~uvex_transients.transients.supernovae.TypeIcSNe`, pairing
      :class:`~uvex_transients.models.supernovae.Ibc.TypeIcSED` with the rate/duration metadata
      described below.

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
           - :math:`R_\mathrm{CC}(z) = k h^2 \psi_\mathrm{UV}(z)`; Type Ic 12.5% of :math:`R_\mathrm{CC}(z)`
           - :footcite:t:`strolger2015`, :footcite:t:`madau2014`,
             :footcite:t:`shivvers2017`
           - Tracks the cosmic star-formation history. Stripped-envelope SNe are 30.4% of core-collapse
             SNe :footcite:p:`shivvers2017`, of which 41.1% are Type Ic :footcite:p:`shivvers2017`, so the Type Ic fraction is
             :math:`0.304 \times 0.411 = 0.125`.
         * - Redshift limit
           - :math:`z = 0.5`
           - --
           - Below the Type IIP and IIb limits, since these events peak at lower luminosity and are cool
             in the UV: in the observability check below, no simulated event beyond
             :math:`z \approx 0.4` clears the UVEX limit.
         * - Duration
           - 100 days
           - --
           - Covers the rise, peak (:math:`t_p \approx 20` d after explosion) and the decline.

      SED Model
      ----------

      :class:`~uvex_transients.models.supernovae.Ibc.TypeIcSED` is a single Bazin pulse times the
      same single-power-law cooling blackbody photosphere used elsewhere in this package (e.g.
      :class:`~uvex_transients.models.supernovae.VillarCoolingBlackbodySED`):

      .. math::

          L_\mathrm{bol}(t) =
          A\,
          \frac{\exp[-(t-t_0)/\tau_\mathrm{fall}]}{1 + \exp[-(t-t_0)/\tau_\mathrm{rise}]},
          \qquad
          T(t) = T_\mathrm{floor} + (T_0 - T_\mathrm{floor})\left(1 + \frac{t}{\tau_T}\right)^{-\alpha_T}.

      The light curve is delegated to
      :class:`~uvex_transients.models.lightcurves.generic.BazinLightcurve`, with the logistic rise
      timescale tied to the transition time, :math:`\tau_\mathrm{rise} = t_0/2.5`. This removes one free
      parameter and, since the rise shape then scales with :math:`t_0`, lets :math:`t_0` alone set the
      time to peak,

      .. math::

          t_p = t_0\left[1 + 0.4\ln\!\left(\frac{2.5\,\tau_\mathrm{fall}}{t_0} - 1\right)\right]
          \approx 1.9\,t_0,

      and leaves the luminosity at :math:`t = 0` at about 14% of the peak. The amplitude :math:`A` is
      a normalization, not the peak luminosity. The priors on :math:`t_0` and :math:`\tau_\mathrm{fall}`
      give a time to peak of about 20 days (5th to 95th percentile 17.8--21.9 d).

      .. dropdown:: Parameter priors

         ``alpha_T`` is held fixed. Fits of the cooling law to the observed temperatures prefer a
         fast, close-to-exponential relaxation to a floor, i.e. a large :math:`\alpha_T` with a long
         :math:`\tau_T`, which are strongly degenerate with the temperature data starting 5--15 days
         after explosion, so one of the two is fixed and the other carries the scatter.

         .. list-table::
            :header-rows: 1
            :widths: 16 12 26 46

            * - Parameter
              - Symbol
              - Prior
              - Notes
            * - ``amplitude``
              - :math:`A`
              - Normal(:math:`\log_{10}(A/\mathrm{erg\,s^{-1}})`; mean=42.6, :math:`\sigma`\=0.3)
              - Bazin normalization; induces a peak :math:`\log_{10} L_p \approx 42.45`.
            * - ``t0``
              - :math:`t_0`
              - Uniform(9.5 d, 12 d)
              - Transition time of the pulse; also sets the rise timescale, :math:`\tau_\mathrm{rise} = t_0/2.5`.
            * - ``fall``
              - :math:`\tau_\mathrm{fall}`
              - Uniform(30 d, 50 d)
              - Exponential decline timescale.
            * - ``T0``
              - :math:`T_0`
              - Normal(:math:`\log_{10}(T_0/\mathrm{K})`; mean=4.2, :math:`\sigma`\=0.1)
              - Photospheric temperature as :math:`t \to 0`.
            * - ``T_floor``
              - :math:`T_\mathrm{floor}`
              - Uniform(4000 K, 5200 K)
              - Asymptotic late-time photospheric temperature.
            * - ``tau_T``
              - :math:`\tau_T`
              - Uniform(15 d, 45 d)
              - Photospheric cooling timescale.
            * - ``alpha_T``
              - :math:`\alpha_T`
              - Fixed (4)
              - Photospheric cooling power-law index.

      Simulated Light Curves
      ~~~~~~~~~~~~~~~~~~~~~~~

      The plot below draws 300 random parameter realizations from the priors above. The top panel shows
      the bolometric light curves, each shifted so that its own peak sits at zero, overlaid with the
      individual bolometric light curves of the Type Ic SNe in :footcite:t:`lyman2016`, which are measured
      relative to maximum light; this checks the *shape* of the light curve (rise, peak width and
      decline). The bottom panel shows the photospheric temperature against time since explosion,
      overlaid with the temperatures of the Type Ic SNe in :footcite:t:`prentice2019`, shifted to time
      since explosion using the tabulated :math:`t_p`.

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

      Observability Summary
      ----------------------

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

         rng = np.random.default_rng(20260918)
         n_samples = 1000

         sn = TypeIcSNe()
         redshift = sn.sample_event_redshift(n_samples, rng=rng)
         params = sn.sed.sample_parameters(size=n_samples, rng=rng)
         params_grid = {pname: value[:, None] for pname, value in params.items()}

         # Convert the integrated rate per steradian to an all-sky rate.
         all_sky_rate = 4 * np.pi * sn.integrated_event_rate * u.sr

         t_grid_rest = np.geomspace(0.1, 100, 300) * u.day
         t_obs_grid = t_grid_rest[None, :] * (1.0 + redshift)[:, None]
         z_grid_bcast = np.broadcast_to(redshift[:, None], t_obs_grid.shape)

         visible_rates = {}
         for band_name, bandpass in uvex.detector.bandpasses.items():
             mag_curve = sn.sed.mag_bandpass(
                 bandpass,
                 t_obs_grid,
                 redshift=z_grid_bcast,
                 **params_grid,
             ).to_value(u.ABmag)
             magnitudes = np.nanmin(mag_curve, axis=1)

             visible = magnitudes < 24.5
             visible_fraction = np.mean(visible)
             visible_rate = visible_fraction * all_sky_rate

             visible_rates[band_name] = visible_rate.to_value(1 / u.yr)

             print(
                 f"{band_name}: {visible_rate:.2f} "
                 f"({visible_fraction:.1%} of events visible)"
             )

         fig, ax = plt.subplots(figsize=(5, 4))
         ax.bar(list(visible_rates), list(visible_rates.values()), color=["C0", "C1"])
         ax.set_yscale("log")
         ax.set_ylabel(r"All-sky rate [yr$^{-1}$]")
         ax.set_title("Peak-visible Type Ic rate")

         fig.tight_layout()

References
-----------

.. footbibliography::
