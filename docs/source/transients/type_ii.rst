.. _transients_type_ii:

Type II Supernovae
====================

Type II core-collapse supernovae (CCSNe) are the explosions of massive stars that retained at
least part of their hydrogen envelope. Type IIP events are the explosions of hydrogen-rich stars: an
early cooling-phase decline (from the initial shock breakout) settles onto a
weeks-to-months-long luminosity "plateau" as a recombination front recedes through the ejecta,
followed by a radioactive tail powered by :math:`^{56}\mathrm{Co}` decay. A subset show a brighter,
hotter, faster early excess on top of this, attributed to shock breakout through and/or collisional
heating of close circumstellar material -- IXF/GGI-like objects, after the prototypes SN 2023ixf
and SN 2024ggi. Both IIP variants share the same SED functional form and differ only in their
parameter priors. Type IIb events, which lost most but not all of their hydrogen, are modeled with
a different, double-pulse form. Each is implemented as its own transient population in a tab below.
(Hydrogen-free Type Ib and Ic supernovae are covered on the :ref:`Type I page <transients_type_i>`.)

.. tab-set::

   .. tab-item:: Type IIP

      Implemented by :class:`~uvex_transients.transients.supernovae.TypeIIPSNe`, pairing
      :class:`~uvex_transients.models.supernovae.IIp.TypeIIPSED` with the rate/duration metadata
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
           - :math:`R_\mathrm{CC}(z) = k\,\psi_\mathrm{UV}(z)`; Type IIP 48.7% of :math:`R_\mathrm{CC}(z)`
           - :footcite:t:`strolger2015`, :footcite:t:`madau2014`, :footcite:t:`li2011`,
             :footcite:t:`shivvers2017`
           - Tracks the cosmic star-formation history. :footcite:t:`li2011` find II-P is
             :math:`69.9^{+5.1}_{-5.8}\%` of the *Type II* rate, which is itself
             :math:`69.6\pm6.7\%` of the total core-collapse rate :footcite:p:`shivvers2017`, so
             the Type IIP fraction of the CC rate is :math:`0.699\times0.696=0.487`. This chain,
             combined in quadrature with the :math:`+27\%/-31\%` uncertainty on
             :footcite:t:`strolger2015`'s :math:`k`, gives
             :attr:`~uvex_transients.transients.supernovae.TypeIIPSNe.RATE_CI`; see
             :ref:`user_guide_transients_rate_uncertainty`.
         * - Redshift limit
           - :math:`z = 0.8`
           - --
           - Matched to ordinary Type IIP peak luminosities.
         * - Duration
           - 100 days
           - --
           - Covers the plateau and the transition to the (unmodeled) nebular phase.

      .. _transients_type_ii_sed:

      .. rubric:: SED Model

      :class:`~uvex_transients.models.supernovae.IIp.TypeIIPSED` is two shared-onset exponential
      components (an early cooling-phase decline onto a constant plateau) plus a switched
      radioactive tail, together with a smooth, doubly-broken power-law photospheric temperature
      that reuses the light curve's own :math:`t_0`/:math:`t_P`:

      .. math::

          L_\mathrm{bol}(t) = S_0(t)\,\bigl[1-S_P(t)\bigr]
          \Bigl[L_\mathrm{pk}\,e^{-(t-t_0)/\tau_\mathrm{cool}} + L_p\Bigr]
          + S_P(t)\, L_\mathrm{Co}\, e^{-(t-t_P)/\tau_\mathrm{Co}},

      with two logistic switches

      .. math::

          S_0(t) = \frac{1}{1+e^{-(t-t_0)/\tau_\mathrm{rise}}}, \qquad
          S_P(t) = \frac{1}{1+e^{-(t-t_P)/\tau_\mathrm{drop}}}.

      :math:`S_0` turns the photospheric emission on at the explosion/rise epoch :math:`t_0`;
      :math:`S_P` turns it back off -- and the radioactive tail on -- at the plateau-end epoch
      :math:`t_P`. Between the rise and :math:`t_P`, :math:`L_\mathrm{bol}` is an early
      cooling-phase decline from :math:`L_\mathrm{pk}` onto a constant plateau :math:`L_p`; after
      :math:`t_P` it switches to a pure radioactive-tail exponential.

      The temperature law has three regimes -- an early regime, a cooling-phase regime around
      :math:`t_0`, and a recombination/plateau regime around :math:`t_P` -- joined by two smooth
      breaks:

      .. math::

          T(t) = T_0\, \left(\frac{t}{t_0}\right)^{\alpha_r}
          \left(\frac{1 + (t/t_0)^{s_0}}{2}\right)^{\frac{\alpha_c - \alpha_r}{s_0}}
          \left(\frac{1 + (t/\sqrt{t_0 t_P})^{s_1}}{2}\right)^{-\frac{\alpha_c - \alpha_p}{s_1}},

      with :math:`t` in days, so that :math:`T(t_0) \approx T_0`. With
      :math:`\alpha_r > 0 > \alpha_c > \alpha_p`, the temperature rises through the early,
      pre-:math:`t_0` region (where :math:`L_\mathrm{bol} \approx 0` anyway), peaks near
      :math:`t_0`, then declines -- fast through the cooling regime and much more slowly through
      the recombination/plateau regime. :math:`s_0`/:math:`s_1` set the sharpness of the two
      breaks. The second break has no epoch of its own, so it is centered on the geometric mean of
      :math:`t_0` and :math:`t_P`.

      .. dropdown:: Parameter priors

         Light-curve priors (``t0`` through ``tau_Co``) come from fitting the light-curve model to
         SN 1999em and SN 2003hn; temperature-law priors (``T_0`` through ``s_1``) are hand-tuned
         against the aggregate Type IIP temperature dataset in the plot below.

         .. list-table::
            :header-rows: 1
            :widths: 16 26 42 16

            * - Parameter
              - Symbol
              - Prior
              - Notes
            * - ``t0``
              - :math:`t_0`
              - Uniform(4 d, 20 d)
              - Explosion/rise reference epoch.
            * - ``tau_rise``
              - :math:`\tau_\mathrm{rise}`
              - Normal(:math:`\log_{10}(\tau_\mathrm{rise}/\mathrm{d})`; mean=0, :math:`\sigma`\=0.2)
              - Rise timescale (median 1 d).
            * - ``L_pk``
              - :math:`L_\mathrm{pk}`
              - Normal(:math:`\log_{10}(L_\mathrm{pk}/\mathrm{erg\,s^{-1}})`; mean=42.48, :math:`\sigma`\=0.2)
              - Early cooling-phase peak luminosity scale (median :math:`\sim3\times10^{42}` erg/s).
            * - ``tau_cool``
              - :math:`\tau_\mathrm{cool}`
              - Uniform(4 d, 10 d)
              - Early cooling-phase decay timescale.
            * - ``L_p``
              - :math:`L_p`
              - Normal(:math:`\log_{10}(L_p/\mathrm{erg\,s^{-1}})`; mean=42.1, :math:`\sigma`\=0.1)
              - Plateau luminosity.
            * - ``t_P``
              - :math:`t_P`
              - Uniform(50 d, 150 d)
              - Plateau-end / radioactive-tail-onset epoch.
            * - ``tau_drop``
              - :math:`\tau_\mathrm{drop}`
              - Fixed (10.5 d)
              - Plateau-end transition width.
            * - ``L_Co``
              - :math:`L_\mathrm{Co}`
              - Normal(:math:`\log_{10}(L_\mathrm{Co}/\mathrm{erg\,s^{-1}})`; mean=41.39, :math:`\sigma`\=0.1)
              - Radioactive-tail luminosity at ``t_P``.
            * - ``tau_Co``
              - :math:`\tau_\mathrm{Co}`
              - Fixed (111.4 d)
              - Radioactive-tail decay timescale, fixed to the :math:`^{56}\mathrm{Co}` e-folding
                time.
            * - ``T_0``
              - :math:`T_0`
              - Normal(:math:`\log_{10}(T_0/\mathrm{K})`; mean=4.05, :math:`\sigma`\=0.1)
              - Normalization: :math:`T(t_0) \approx T_0` (:math:`\sim1.1\times10^4` K).
            * - ``alpha_r``
              - :math:`\alpha_r`
              - Normal(2, :math:`\sigma`\=0.2)
              - Early-regime power-law index; :math:`T` rises toward :math:`t_0`.
            * - ``alpha_c``
              - :math:`\alpha_c`
              - Normal(-0.45, :math:`\sigma`\=0.1)
              - Cooling-regime power-law index; :math:`T` declines.
            * - ``alpha_p``
              - :math:`\alpha_p`
              - Normal(-0.1, :math:`\sigma`\=0.01)
              - Plateau-regime power-law index; :math:`T` declines slowly.
            * - ``s_0``
              - :math:`s_0`
              - Uniform(10, 20)
              - Sharpness of the break at :math:`t_0`.
            * - ``s_1``
              - :math:`s_1`
              - Uniform(10, 50)
              - Sharpness of the break at :math:`\sqrt{t_0 t_P}`.

      .. rubric:: Simulated Light Curves

      The plot below draws 300 random parameter realizations from the priors above and shows the
      resulting bolometric light curves and photospheric temperatures, overlaid with the observed
      light curves and temperatures of several Type IIP SNe :footcite:p:`dallora2014,faran2018`.

      .. plot::
         :include-source: false

         import numpy as np
         import matplotlib.pyplot as plt
         from astropy import units as u

         from uvex_transients.models.supernovae import TypeIIPSED
         from uvex_transients.utils.lightcurve_archive import LightcurveArchive

         rng = np.random.default_rng(20260910)
         n_samples = 300

         params = TypeIIPSED().sample_parameters(size=n_samples, rng=rng)
         params_grid = {name: value[:, None] for name, value in params.items()}

         t = np.geomspace(0.5, 200, 400) * u.day
         L_bol = TypeIIPSED.eval_bolometric(t, **params_grid)
         T = TypeIIPSED.temperature(t, **params_grid)

         fig, (ax_L, ax_T) = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True)

         for row in range(n_samples):
             ax_L.plot(t.to_value(u.day), L_bol[row].to_value(u.erg / u.s), color="C0", lw=0.4, alpha=0.2)
             ax_T.plot(t.to_value(u.day), T[row].to_value(u.K), color="C3", lw=0.4, alpha=0.2)

         archive = LightcurveArchive()
         observed_iip_sne = [
             ("1999em_bersten2009", "SN 1999em", "o", "k"),
             ("2003hn_bersten2009", "SN 2003hn", "s", "firebrick"),
             ("2012aw_dallora14", "SN 2012aw", "^", "C2"),
             ("2012A_faran18", "SN 2012A", "X", "C7"),
             ("2008in_faran18", "SN 2008in", "*", "C8"),
         ]
         # Light-curve-only comparison objects (no photospheric temperature sequence available).
         lbol_only_iip_sne = [
             ("2004et_dallora14", "SN 2004et", "v", "C4"),
             ("1992H_dallora14", "SN 1992H", "D", "C5"),
             ("2009bw_dallora14", "SN 2009bw", "P", "C6"),
         ]

         for suffix, label, marker, color in observed_iip_sne:
             lbol_obs = archive.table("supernovae/IIP", suffix, "L_bol")
             Tphot_obs = archive.table("supernovae/IIP", suffix, "T_phot")
             ax_L.scatter(
                 lbol_obs["time"].to_value(u.day), lbol_obs["L_bol"].to_value(u.erg / u.s),
                 marker=marker, s=28, color=color, edgecolor="white", linewidth=0.5, zorder=5,
                 label=label,
             )
             ax_T.scatter(
                 Tphot_obs["time"].to_value(u.day), Tphot_obs["T_phot"].to_value(u.K),
                 marker=marker, s=28, color=color, edgecolor="white", linewidth=0.5, zorder=5,
                 label=label,
             )

         for suffix, label, marker, color in lbol_only_iip_sne:
             lbol_obs = archive.table("supernovae/IIP", suffix, "L_bol")
             ax_L.scatter(
                 lbol_obs["time"].to_value(u.day), lbol_obs["L_bol"].to_value(u.erg / u.s),
                 marker=marker, s=28, color=color, edgecolor="white", linewidth=0.5, zorder=5,
                 label=label,
             )

         ax_L.set_xscale("log")
         ax_L.set_yscale("log")
         ax_L.set_ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
         ax_L.set_title("Type IIP SNe: simulated bolometric light curves (n=300)")
         ax_L.legend(loc="lower left", fontsize=8, frameon=False)
         ax_L.set_ylim([1e40,None])
         ax_T.set_yscale("log")
         ax_T.set_xlabel("Time since explosion [days]")
         ax_T.set_ylabel("Photospheric temperature [K]")
         ax_T.set_ylim([1e3,None])

         fig.tight_layout()

      .. rubric:: Observability Summary

      Below are the redshifts :math:`z` and corresponding bandpass calculated peak apparent AB
      magnitudes :math:`m_\mathrm{AB}` of 3000 simulated events drawn from the priors above, with
      the UVEX 1 Dwell limit of :math:`m<24.5` overplotted. Because the bolometric light curve does
      not peak exactly at a single named parameter, the peak apparent magnitude is found by a
      numerical search over each event's light curve rather than evaluated at a fixed time.

      .. plot::
         :include-source: false

          import numpy as np
          import matplotlib.pyplot as plt
          from astropy import units as u
          from scipy.stats import gaussian_kde

          from m4opt.missions import uvex
          from uvex_transients.transients.supernovae import TypeIIPSNe
          from uvex_transients.models.supernovae import TypeIIPSED

          rng = np.random.default_rng(20260911)
          n_samples = 3000

          sn = TypeIIPSNe()
          z = sn.sample_event_redshift(n_samples, rng=rng)
          params = TypeIIPSED().sample_parameters(size=n_samples, rng=rng)
          params_grid = {name: value[:, None] for name, value in params.items()}

          # Numerically search each event's own light curve for its brightest (peak) apparent
          # magnitude, rather than assuming a single named parameter marks the true peak.
          t_grid_rest = np.geomspace(0.5, 200, 200) * u.day
          t_obs_grid = t_grid_rest[None, :] * (1.0 + z)[:, None]
          z_grid_bcast = np.broadcast_to(z[:, None], t_obs_grid.shape)

          bandpasses = uvex.detector.bandpasses
          band_names = list(bandpasses)

          fig, axes = plt.subplots(1, len(band_names), figsize=(10.5, 4.8), sharey=True)

          for ax, band_name in zip(axes, band_names):
              mag_curve = TypeIIPSED.mag_bandpass(
                  bandpasses[band_name], t_obs_grid, redshift=z_grid_bcast, **params_grid
              ).to_value(u.ABmag)
              mag = np.nanmin(mag_curve, axis=1)
              finite = np.isfinite(mag)
              z_finite, mag_finite = z[finite], mag[finite]

              ax.scatter(z_finite, mag_finite, s=5, ec='k', fc='k', alpha=0.5, label="Simulated events")

              kde = gaussian_kde(np.vstack([z_finite, mag_finite]))
              z_kde_grid = np.linspace(z_finite.min(), z_finite.max(), 150)
              mag_kde_grid = np.linspace(mag_finite.min(), mag_finite.max(), 150)
              Z_grid, Mag_grid = np.meshgrid(z_kde_grid, mag_kde_grid)
              density = kde(np.vstack([Z_grid.ravel(), Mag_grid.ravel()])).reshape(Z_grid.shape)
              ax.contour(Z_grid, Mag_grid, density, levels=6, colors="k", linewidths=0.7)

              ax.axhline(24.5, color="firebrick", ls="--", lw=1.2, label="UVEX limit (1 Dwell)")

              ax.invert_yaxis()
              ax.set_xlabel("Redshift")
              ax.set_title(f"UVEX {band_name}")
              ax.legend(loc="upper right", fontsize=8, frameon=False)

              ax.invert_yaxis()
              ax.set_ylim([35, 15])

          axes[0].set_ylabel("Peak apparent AB magnitude")
          fig.suptitle(f"Type IIP SNe: peak apparent magnitude vs. redshift (n={n_samples})")
          fig.tight_layout()
          plt.show()

      The anticipated rate detectable by UVEX at these limits is as follows, assuming that any
      event above the :math:`m<24.5` limit is detectable, and that the population is isotropic and
      homogeneous in comoving volume out to its redshift limit:

      .. plot::
         :include-source: false

          import numpy as np
          import matplotlib.pyplot as plt
          from astropy import units as u

          from m4opt.missions import uvex
          from uvex_transients.transients.supernovae import TypeIIPSNe

          rng = np.random.default_rng(20260911)
          n_samples = 3000

          sn = TypeIIPSNe()
          redshift = sn.sample_event_redshift(n_samples, rng=rng)
          params = sn.sed.sample_parameters(size=n_samples, rng=rng)
          params_grid = {pname: value[:, None] for pname, value in params.items()}

          # The all-sky rate, with no survey footprint applied.
          all_sky_rate = sn.all_sky_rate

          # Numerically search each event's own light curve for its brightest (peak)
          # apparent magnitude -- see the discussion above for why the peak
          # doesn't sit at a single named parameter.
          t_grid_rest = np.geomspace(0.5, 200, 200) * u.day
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

          # Plot all-sky visible rates, by band.
          fig, ax = plt.subplots(figsize=(5, 4))
          ax.bar(list(visible_rates), list(visible_rates.values()), color=["C0", "C1"])
          ax.set_yscale("log")
          ax.set_ylabel(r"All-sky rate [yr$^{-1}$]")
          ax.set_title("Peak-visible Type IIP SNe rate")

          fig.tight_layout()
          plt.show()

   .. tab-item:: Type IIP + Excess

      Implemented by :class:`~uvex_transients.transients.supernovae.TypeIIPExcessSNe`, pairing
      :class:`~uvex_transients.models.supernovae.IIp.TypeIIPExcessSED` with the rate/duration
      metadata described below. IXF/GGI-like: a bright, hot, fast early excess on top of the same
      ordinary Type IIP evolution, attributed to shock breakout through and/or collisional heating
      of close circumstellar material.

      .. rubric:: Quick Facts

      .. list-table::
         :header-rows: 1
         :widths: 15 25 15 45

         * - Quantity
           - Value
           - Source
           - Notes
         * - Rate
           - 30% of the Type IIP rate (14.6% of :math:`R_\mathrm{CC}(z)`)
           - :footcite:t:`bruch2023`
           - Reflects the high incidence of early CSM-interaction signatures found among Type II
             SNe (ZTF). The 30% multiplier has no published uncertainty of its own, so
             :attr:`~uvex_transients.transients.supernovae.TypeIIPExcessSNe.RATE_CI` carries
             exactly the same relative uncertainty as ordinary Type IIP's.
         * - Redshift limit
           - :math:`z = 2`
           - --
           - Higher than ordinary Type IIP, to admit this population's brighter, more UV-luminous
             early-cooling realizations.
         * - Duration
           - 100 days
           - --
           - Same window as ordinary Type IIP.

      .. rubric:: SED Model

      :class:`~uvex_transients.models.supernovae.IIp.TypeIIPExcessSED` is a subclass of
      :class:`~uvex_transients.models.supernovae.IIp.TypeIIPSED`: the same light curve and
      temperature functional forms,

      .. math::

          L_\mathrm{bol}(t) = S_0(t)\,\bigl[1-S_P(t)\bigr]
          \Bigl[L_\mathrm{pk}\,e^{-(t-t_0)/\tau_\mathrm{cool}} + L_p\Bigr]
          + S_P(t)\, L_\mathrm{Co}\, e^{-(t-t_P)/\tau_\mathrm{Co}},
          \qquad
          T(t) = T_0\, \left(\frac{t}{t_0}\right)^{\alpha_r}
          \left(\frac{1 + (t/t_0)^{s_0}}{2}\right)^{\frac{\alpha_c - \alpha_r}{s_0}}
          \left(\frac{1 + (t/\sqrt{t_0 t_P})^{s_1}}{2}\right)^{-\frac{\alpha_c - \alpha_p}{s_1}},

      only with different parameter priors -- an earlier, faster rise (``t0``, ``tau_rise``), a
      higher peak luminosity (``L_pk``) and normalization temperature (``T_0``), and a shallower
      early-regime temperature index (``alpha_r``) -- shifted to match the brighter, hotter, faster
      early cooling phase seen in SN 2023ixf and SN 2024ggi, rather than the smoother early decline
      of ordinary Type IIP SNe.

      .. dropdown:: Parameter priors

         Hand-tuned against SN 2023ixf :footcite:p:`hsu2025` and SN 2024ggi
         :footcite:p:`chen2024`, the only two objects of this kind currently in
         ``test_data/transients``. Parameters not listed below (``tau_drop``, ``tau_Co``) are
         fixed to the same values as ordinary Type IIP.

         .. list-table::
            :header-rows: 1
            :widths: 16 26 42 16

            * - Parameter
              - Symbol
              - Prior
              - Notes
            * - ``t0``
              - :math:`t_0`
              - Uniform(2 d, 5 d)
              - Explosion/rise reference epoch.
            * - ``tau_rise``
              - :math:`\tau_\mathrm{rise}`
              - Normal(:math:`\log_{10}(\tau_\mathrm{rise}/\mathrm{d})`; mean=-0.5, :math:`\sigma`\=0.2)
              - Rise timescale (median :math:`\sim0.3` d).
            * - ``L_pk``
              - :math:`L_\mathrm{pk}`
              - Normal(:math:`\log_{10}(L_\mathrm{pk}/\mathrm{erg\,s^{-1}})`; mean=43.3, :math:`\sigma`\=0.25)
              - Early cooling-phase peak luminosity scale (median :math:`\sim2\times10^{43}` erg/s).
            * - ``tau_cool``
              - :math:`\tau_\mathrm{cool}`
              - Uniform(4 d, 15 d)
              - Early cooling-phase decay timescale.
            * - ``L_p``
              - :math:`L_p`
              - Normal(:math:`\log_{10}(L_p/\mathrm{erg\,s^{-1}})`; mean=42.2, :math:`\sigma`\=0.2)
              - Plateau luminosity.
            * - ``t_P``
              - :math:`t_P`
              - Uniform(50 d, 150 d)
              - Plateau-end / radioactive-tail-onset epoch.
            * - ``L_Co``
              - :math:`L_\mathrm{Co}`
              - Normal(:math:`\log_{10}(L_\mathrm{Co}/\mathrm{erg\,s^{-1}})`; mean=41.5, :math:`\sigma`\=0.3)
              - Radioactive-tail luminosity at ``t_P``.
            * - ``T_0``
              - :math:`T_0`
              - Normal(:math:`\log_{10}(T_0/\mathrm{K})`; mean=4.2, :math:`\sigma`\=0.1)
              - Normalization: :math:`T(t_0) \approx T_0` (:math:`\sim1.6\times10^4` K).
            * - ``alpha_r``
              - :math:`\alpha_r`
              - Normal(1, :math:`\sigma`\=0.2)
              - Early-regime power-law index; :math:`T` rises toward :math:`t_0`.
            * - ``alpha_c``
              - :math:`\alpha_c`
              - Normal(-0.45, :math:`\sigma`\=0.1)
              - Cooling-regime power-law index; :math:`T` declines.
            * - ``alpha_p``
              - :math:`\alpha_p`
              - Normal(-0.1, :math:`\sigma`\=0.01)
              - Plateau-regime power-law index; :math:`T` declines slowly.
            * - ``s_0``
              - :math:`s_0`
              - Uniform(10, 20)
              - Sharpness of the break at :math:`t_0`.
            * - ``s_1``
              - :math:`s_1`
              - Uniform(10, 50)
              - Sharpness of the break at :math:`\sqrt{t_0 t_P}`.

      .. rubric:: Simulated Light Curves

      The plot below draws 300 random parameter realizations from the priors above and shows the
      resulting bolometric light curves and photospheric temperatures, overlaid with the observed
      light curves and temperatures of SN 2023ixf :footcite:p:`hsu2025` and SN 2024ggi
      :footcite:p:`chen2024`.

      .. plot::
         :include-source: false

         import numpy as np
         import matplotlib.pyplot as plt
         from astropy import units as u

         from uvex_transients.models.supernovae import TypeIIPExcessSED
         from uvex_transients.utils.lightcurve_archive import LightcurveArchive

         rng = np.random.default_rng(20260910)
         n_samples = 300

         params = TypeIIPExcessSED().sample_parameters(size=n_samples, rng=rng)
         params_grid = {name: value[:, None] for name, value in params.items()}

         t = np.geomspace(0.2, 200, 400) * u.day
         L_bol = TypeIIPExcessSED.eval_bolometric(t, **params_grid)
         T = TypeIIPExcessSED.temperature(t, **params_grid)

         fig, (ax_L, ax_T) = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True)

         for row in range(n_samples):
             ax_L.plot(t.to_value(u.day), L_bol[row].to_value(u.erg / u.s), color="C0", lw=0.4, alpha=0.2)
             ax_T.plot(t.to_value(u.day), T[row].to_value(u.K), color="C3", lw=0.4, alpha=0.2)

         archive = LightcurveArchive()
         observed_excess_sne = [
             ("2023ixf_hsu2025", "SN 2023ixf", "o", "k"),
             ("2024ggi_chen2024", "SN 2024ggi", "s", "firebrick"),
         ]

         for suffix, label, marker, color in observed_excess_sne:
             lbol_obs = archive.table("supernovae/II", suffix, "L_bol")
             Tphot_obs = archive.table("supernovae/II", suffix, "T_phot")
             ax_L.scatter(
                 lbol_obs["time"].to_value(u.day), lbol_obs["L_bol"].to_value(u.erg / u.s),
                 marker=marker, s=28, color=color, edgecolor="white", linewidth=0.5, zorder=5,
                 label=label,
             )
             ax_T.scatter(
                 Tphot_obs["time"].to_value(u.day), Tphot_obs["T_phot"].to_value(u.K),
                 marker=marker, s=28, color=color, edgecolor="white", linewidth=0.5, zorder=5,
                 label=label,
             )

         ax_L.set_xscale("log")
         ax_L.set_yscale("log")
         ax_L.set_ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
         ax_L.set_title("Type IIP + Excess SNe: simulated bolometric light curves (n=300)")
         ax_L.legend(loc="lower left", fontsize=8, frameon=False)
         ax_L.set_ylim([1e40,None])
         ax_T.set_yscale("log")
         ax_T.set_xlabel("Time since explosion [days]")
         ax_T.set_ylabel("Photospheric temperature [K]")
         ax_T.set_ylim([1e3,None])

         fig.tight_layout()

      .. rubric:: Observability Summary

      Below are the redshifts :math:`z` and corresponding bandpass calculated peak apparent AB
      magnitudes :math:`m_\mathrm{AB}` of 3000 simulated events drawn from the priors above, with
      the UVEX 1 Dwell limit of :math:`m<24.5` overplotted.

      .. plot::
         :include-source: false

          import numpy as np
          import matplotlib.pyplot as plt
          from astropy import units as u
          from scipy.stats import gaussian_kde

          from m4opt.missions import uvex
          from uvex_transients.transients.supernovae import TypeIIPExcessSNe
          from uvex_transients.models.supernovae import TypeIIPExcessSED

          rng = np.random.default_rng(20260911)
          n_samples = 3000

          sn = TypeIIPExcessSNe()
          z = sn.sample_event_redshift(n_samples, rng=rng)
          params = TypeIIPExcessSED().sample_parameters(size=n_samples, rng=rng)
          params_grid = {name: value[:, None] for name, value in params.items()}

          # Numerically search each event's own light curve for its brightest (peak) apparent
          # magnitude, rather than assuming a single named parameter marks the true peak.
          t_grid_rest = np.geomspace(0.2, 200, 200) * u.day
          t_obs_grid = t_grid_rest[None, :] * (1.0 + z)[:, None]
          z_grid_bcast = np.broadcast_to(z[:, None], t_obs_grid.shape)

          bandpasses = uvex.detector.bandpasses
          band_names = list(bandpasses)

          fig, axes = plt.subplots(1, len(band_names), figsize=(10.5, 4.8), sharey=True)

          for ax, band_name in zip(axes, band_names):
              mag_curve = TypeIIPExcessSED.mag_bandpass(
                  bandpasses[band_name], t_obs_grid, redshift=z_grid_bcast, **params_grid
              ).to_value(u.ABmag)
              mag = np.nanmin(mag_curve, axis=1)
              finite = np.isfinite(mag)
              z_finite, mag_finite = z[finite], mag[finite]

              ax.scatter(z_finite, mag_finite, s=5, ec='k', fc='k', alpha=0.5, label="Simulated events")

              kde = gaussian_kde(np.vstack([z_finite, mag_finite]))
              z_kde_grid = np.linspace(z_finite.min(), z_finite.max(), 150)
              mag_kde_grid = np.linspace(mag_finite.min(), mag_finite.max(), 150)
              Z_grid, Mag_grid = np.meshgrid(z_kde_grid, mag_kde_grid)
              density = kde(np.vstack([Z_grid.ravel(), Mag_grid.ravel()])).reshape(Z_grid.shape)
              ax.contour(Z_grid, Mag_grid, density, levels=6, colors="k", linewidths=0.7)

              ax.axhline(24.5, color="firebrick", ls="--", lw=1.2, label="UVEX limit (1 Dwell)")

              ax.invert_yaxis()
              ax.set_xlabel("Redshift")
              ax.set_title(f"UVEX {band_name}")
              ax.legend(loc="upper right", fontsize=8, frameon=False)

              ax.invert_yaxis()
              ax.set_ylim([35, 15])

          axes[0].set_ylabel("Peak apparent AB magnitude")
          fig.suptitle(f"Type IIP + Excess SNe: peak apparent magnitude vs. redshift (n={n_samples})")
          fig.tight_layout()
          plt.show()

      The anticipated rate detectable by UVEX at these limits is as follows, assuming that any
      event above the :math:`m<24.5` limit is detectable, and that the population is isotropic and
      homogeneous in comoving volume out to its redshift limit:

      .. plot::
         :include-source: false

          import numpy as np
          import matplotlib.pyplot as plt
          from astropy import units as u

          from m4opt.missions import uvex
          from uvex_transients.transients.supernovae import TypeIIPExcessSNe

          rng = np.random.default_rng(20260911)
          n_samples = 3000

          sn = TypeIIPExcessSNe()
          redshift = sn.sample_event_redshift(n_samples, rng=rng)
          params = sn.sed.sample_parameters(size=n_samples, rng=rng)
          params_grid = {pname: value[:, None] for pname, value in params.items()}

          # The all-sky rate, with no survey footprint applied.
          all_sky_rate = sn.all_sky_rate

          # Numerically search each event's own light curve for its brightest (peak)
          # apparent magnitude -- see the discussion above for why the peak
          # doesn't sit at a single named parameter.
          t_grid_rest = np.geomspace(0.2, 200, 200) * u.day
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

          # Plot all-sky visible rates, by band.
          fig, ax = plt.subplots(figsize=(5, 4))
          ax.bar(list(visible_rates), list(visible_rates.values()), color=["C0", "C1"])
          ax.set_yscale("log")
          ax.set_ylabel(r"All-sky rate [yr$^{-1}$]")
          ax.set_title("Peak-visible Type IIP + Excess SNe rate")

          fig.tight_layout()
          plt.show()

   .. tab-item:: Type IIb

      Type IIb supernovae are core-collapse explosions of massive stars that have been stripped of most,
      but not all, of their hydrogen envelope. Many show a double-peaked light curve: an early, hours-
      to-days-long flash powered by the shock heating and subsequent cooling of the extended envelope,
      followed -- after a dip -- by a broader, weeks-long peak powered by radioactive
      :math:`^{56}\mathrm{Ni}` decay, the same mechanism that powers most other core-collapse SN light
      curves. Others show only the single, radioactively powered peak, with no resolved early bump.
      Where :class:`~uvex_transients.transients.supernovae.ShockCoolingIIb` models only the shock-cooling
      component from first principles, this population is a purely phenomenological light curve intended
      to span the whole population -- single- and double-peaked events alike -- in a single functional
      form.

      This population is implemented by
      :class:`~uvex_transients.transients.supernovae.TypeIIbSNe`, pairing
      :class:`~uvex_transients.models.supernovae.IIb.TypeIIbSED` with the rate/duration metadata
      described below.

      .. note::

         The priors below are broad, order-of-magnitude-motivated ranges, not yet a fit to any specific
         real Type IIb event. This page will be updated if/when they are recalibrated against data (as
         :class:`~uvex_transients.models.supernovae.IIp.TypeIIPSED` was).

      .. rubric:: Quick Facts

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
             :math:`30.4^{+5.0}_{-4.9}\%` of the total core-collapse rate, so the Type IIb fraction
             of the CC rate is :math:`0.340\times0.304=0.103`. Combined in quadrature with
             :footcite:t:`strolger2015`'s :math:`+27\%/-31\%` normalization uncertainty, this gives
             :attr:`~uvex_transients.transients.supernovae.TypeIIbSNe.RATE_CI` (see
             :ref:`user_guide_transients_rate_uncertainty`). Identical rate to
             :class:`~uvex_transients.transients.supernovae.ShockCoolingIIb` -- both describe the
             same underlying Type IIb population, just with different SED models.
         * - Redshift limit
           - :math:`z = 0.5`
           - --
           - Tighter than the :math:`z = 1` bound of :class:`~uvex_transients.transients.supernovae.ShockCoolingIIb`.
         * - Duration
           - 200 days
           - --
           - Long enough to cover the shock-cooling peak (where present), the dip, the radioactive main
             peak, and its subsequent decline -- unlike
             :class:`~uvex_transients.transients.supernovae.ShockCoolingIIb`'s much shorter 20 day
             window, which covers only the first of those phases.

      .. rubric:: SED Model

      :class:`~uvex_transients.models.supernovae.IIb.TypeIIbSED` pairs a superposition of two Bazin
      pulses with the same single-power-law cooling blackbody photosphere used elsewhere in this package
      (e.g. :class:`~uvex_transients.models.supernovae.VillarCoolingBlackbodySED`):

      .. math::

          L_\mathrm{bol}(t) =
          A_0\,
          \frac{\exp[-(t-t_0)/\tau_{\mathrm{fall},0}]}{1 + \exp[-(t-t_0)/\tau_{\mathrm{rise},0}]}
          +
          A_1\,
          \frac{\exp[-(t-t_1)/\tau_{\mathrm{fall},1}]}{1 + \exp[-(t-t_1)/\tau_{\mathrm{rise},1}]},
          \qquad
          T(t) = T_\mathrm{floor} + (T_0 - T_\mathrm{floor})\left(1 + \frac{t}{\tau_T}\right)^{-\alpha_T}.

      The light curve is delegated directly to
      :class:`~uvex_transients.models.lightcurves.generic.TwoComponentBazinLightcurve`: two ordinary
      :class:`~uvex_transients.models.lightcurves.generic.BazinLightcurve` pulses, added rather than
      multiplied -- an early one centered on :math:`t_0` standing in for the shock-cooling peak, and a
      later one centered on :math:`t_1` for the radioactively powered main peak. Because the two
      components are independent and additive, a single functional form covers both populations at once:
      with the early component's amplitude :math:`A_0` much smaller than the main peak's :math:`A_1`,
      only the main peak is visible (a single-peaked event); with :math:`A_0` comparable to :math:`A_1`,
      both peaks show, with a dip between them where each pulse has decayed enough for the other to
      dominate (a double-peaked event). This fits real double- and single-peaked Type IIb light curves
      better than a single pulse reshaped by a multiplicative modulation.

      .. dropdown:: Parameter priors

         ``amplitude_0`` -- the early peak's normalization -- is uniform in
         :math:`\log_{10}(A_0/\mathrm{erg\,s^{-1}})` between 39 and 43, i.e. from
         :math:`10^{39}` to :math:`10^{43}\ \mathrm{erg\,s^{-1}}`: far fainter than the main peak (an
         effectively single-peaked draw) up to brighter than it (a double-peaked, or even
         early-peak-dominated, draw); roughly a third of draws from the priors below are double-peaked.
         ``t0`` and ``T_floor`` are held fixed; every other parameter is drawn from a broad Uniform (or,
         for ``amplitude_1``/``T0``, Normal-in-log) prior.

         .. list-table::
            :header-rows: 1
            :widths: 16 12 26 46

            * - Parameter
              - Symbol
              - Prior
              - Notes
            * - ``amplitude_0``
              - :math:`A_0`
              - LogUniform(:math:`10^{39}`, :math:`10^{43}\ \mathrm{erg\,s^{-1}}`)
              - Early, shock-cooling peak normalization; spans negligible to brighter than the main peak.
            * - ``t0``
              - :math:`t_0`
              - Fixed (2 d)
              - Transition time of the early peak.
            * - ``rise_0``
              - :math:`\tau_{\mathrm{rise},0}`
              - Uniform(0.3 d, 1 d)
              - Logistic rise timescale of the early peak.
            * - ``fall_0``
              - :math:`\tau_{\mathrm{fall},0}`
              - Uniform(3 d, 10 d)
              - Exponential decline timescale of the early peak.
            * - ``amplitude_1``
              - :math:`A_1`
              - Normal(:math:`\log_{10}(A_1/\mathrm{erg\,s^{-1}})`; mean=42.5, :math:`\sigma`\=0.1)
              - Main, radioactively powered peak normalization (median :math:`\sim3\times10^{42}` erg/s).
            * - ``t1``
              - :math:`t_1`
              - Uniform(10 d, 20 d)
              - Transition time of the main peak.
            * - ``rise_1``
              - :math:`\tau_{\mathrm{rise},1}`
              - Uniform(2 d, 4 d)
              - Logistic rise timescale of the main peak.
            * - ``fall_1``
              - :math:`\tau_{\mathrm{fall},1}`
              - Uniform(30 d, 55 d)
              - Exponential decline timescale of the main peak.
            * - ``T0``
              - :math:`T_0`
              - Normal(:math:`\log_{10}(T_0/\mathrm{K})`; mean=4.1, :math:`\sigma`\=0.05)
              - Photospheric temperature as :math:`t \to 0` (median :math:`\sim1.3\times10^4` K).
            * - ``T_floor``
              - :math:`T_\mathrm{floor}`
              - Fixed (:math:`4\times10^3` K)
              - Asymptotic late-time photospheric temperature.
            * - ``tau_T``
              - :math:`\tau_T`
              - Uniform(5 d, 10 d)
              - Photospheric cooling timescale.
            * - ``alpha_T``
              - :math:`\alpha_T`
              - Uniform(0.7, 1.3)
              - Photospheric cooling power-law index.

      .. rubric:: Simulated Light Curves

      The plot below draws 300 random parameter realizations from the priors above and shows the
      resulting bolometric light curves and photospheric temperatures, overlaid with the observed
      light curves and temperatures of several Type IIb SNe
      :footcite:p:`richmond1994,taubenberger2011,moralesgaroffolo2014,moralesgaroffolo2015,yamanaka2025,prentice2019`.
      The spread illustrates the single-/double-peaked split described above: most realizations show
      only the main peak, with a minority showing a distinct early bump or, at the high end of the
      ``amplitude_0`` prior, an early-dominated light curve -- both SN 1993J and SN 2011fu are
      well-known double-peaked events, and show up as such in the overlaid data.

      .. plot::
         :include-source: false

         import numpy as np
         import matplotlib.pyplot as plt
         from astropy import units as u

         from uvex_transients.models.supernovae import TypeIIbSED as SEDClass
         from uvex_transients.utils.lightcurve_archive import LightcurveArchive

         rng = np.random.default_rng(20260918)
         n_samples = 300

         params = SEDClass().sample_parameters(size=n_samples, rng=rng)
         params_grid = {name: value[:, None] for name, value in params.items()}

         t = np.geomspace(0.1, 200, 400) * u.day
         L_bol = SEDClass.eval_bolometric(t, **params_grid)
         T = SEDClass.temperature(t, **params_grid)

         fig, (ax_L, ax_T) = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True)

         for row in range(n_samples):
             ax_L.plot(t.to_value(u.day), L_bol[row].to_value(u.erg / u.s), color="C0", lw=0.4, alpha=0.15)
             ax_T.plot(t.to_value(u.day), T[row].to_value(u.K), color="C3", lw=0.4, alpha=0.15)

         archive = LightcurveArchive()
         observed_iib_sne = [
             ("2011fu_moralesgaroffolo2015", "SN 2011fu", "o", "k"),
             ("2013df_moralesgaroffolo2014", "SN 2013df", "s", "firebrick"),
         ]
         # Temperature-only comparison objects (photospheric temperatures from Prentice+19, shifted to
         # time since explosion using the tabulated peak time; no bolometric light curve available).
         tphot_only_iib_sne = [
             ("2013bb_prentice2019", "SN 2013bb", "P", "C1"),
             ("2016gkg_prentice2019", "SN 2016gkg", "h", "C5"),
             ("2017ixz_prentice2019", "SN 2017ixz", "<", "C6"),
         ]
         # Light-curve-only comparison objects (no photospheric temperature sequence available).
         lbol_only_iib_sne = [
             ("1993J_richmond1994", "SN 1993J", "^", "C2"),
             ("2008ax_taubenberger2011", "SN 2008ax", "X", "C7"),
             ("2024iss_yamanaka2025", "SN 2024iss", "*", "C8"),
         ]

         for suffix, label, marker, color in observed_iib_sne:
             lbol_obs = archive.table("supernovae/IIb", suffix, "L_bol")
             Tphot_obs = archive.table("supernovae/IIb", suffix, "T_phot")
             ax_L.scatter(
                 lbol_obs["time"].to_value(u.day), lbol_obs["L_bol"].to_value(u.erg / u.s),
                 marker=marker, s=28, color=color, edgecolor="white", linewidth=0.5, zorder=5,
                 label=label,
             )
             ax_T.scatter(
                 Tphot_obs["time"].to_value(u.day), Tphot_obs["T_phot"].to_value(u.K),
                 marker=marker, s=28, color=color, edgecolor="white", linewidth=0.5, zorder=5,
                 label=label,
             )

         for suffix, label, marker, color in tphot_only_iib_sne:
             Tphot_obs = archive.table("supernovae/IIb", suffix, "T_phot")
             ax_T.scatter(
                 Tphot_obs["time"].to_value(u.day), Tphot_obs["T_phot"].to_value(u.K),
                 marker=marker, s=28, color=color, edgecolor="white", linewidth=0.5, zorder=5,
                 label=label,
             )

         for suffix, label, marker, color in lbol_only_iib_sne:
             lbol_obs = archive.table("supernovae/IIb", suffix, "L_bol")
             ax_L.scatter(
                 lbol_obs["time"].to_value(u.day), lbol_obs["L_bol"].to_value(u.erg / u.s),
                 marker=marker, s=28, color=color, edgecolor="white", linewidth=0.5, zorder=5,
                 label=label,
             )

         ax_L.set_xscale("log")
         ax_L.set_yscale("log")
         ax_L.set_ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
         ax_L.set_title("Type IIb: simulated bolometric light curves (n=300)")
         ax_L.legend(loc="lower left", fontsize=8, frameon=False)
         ax_L.set_ylim([1e39, None])

         ax_T.set_yscale("log")
         ax_T.set_xlabel("Time since explosion [days]")
         ax_T.set_ylabel("Photospheric temperature [K]")
         ax_T.set_ylim([1e3, None])

         fig.tight_layout()



      .. rubric:: Observability Summary

      Below are the redshifts :math:`z` and corresponding bandpass peak apparent AB magnitudes
      :math:`m_\mathrm{AB}` of 1000 simulated events drawn from the priors above, with the UVEX 1 Dwell
      limit of :math:`m<24.5` overplotted. The peak apparent magnitude is found by a numerical search
      over each event's light curve, since neither peak sits at a single named parameter once both
      components contribute.

      .. plot::
         :include-source: false

         import numpy as np
         import matplotlib.pyplot as plt
         from astropy import units as u

         from m4opt.missions import uvex
         from uvex_transients.transients.supernovae import TypeIIbSNe
         from uvex_transients.models.supernovae import TypeIIbSED

         rng = np.random.default_rng(20260918)
         n_samples = 1000

         sn = TypeIIbSNe()
         z = sn.sample_event_redshift(n_samples, rng=rng)
         params = TypeIIbSED().sample_parameters(size=n_samples, rng=rng)
         params_grid = {name: value[:, None] for name, value in params.items()}

         # Numerically search each event's own light curve for its brightest (peak) apparent
         # magnitude, since neither the early nor the main peak sits at a single named parameter.
         t_grid_rest = np.geomspace(0.1, 200, 300) * u.day
         t_obs_grid = t_grid_rest[None, :] * (1.0 + z)[:, None]
         z_grid_bcast = np.broadcast_to(z[:, None], t_obs_grid.shape)

         bandpasses = uvex.detector.bandpasses
         band_names = list(bandpasses)

         fig, axes = plt.subplots(1, len(band_names), figsize=(10.5, 4.8), sharey=True)

         for ax, band_name in zip(axes, band_names):
             mag_curve = TypeIIbSED.mag_bandpass(
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
         fig.suptitle(f"Type IIb: peak apparent magnitude vs. redshift (n={n_samples})")
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
         from uvex_transients.transients.supernovae import TypeIIbSNe

         rng = np.random.default_rng(20260918)
         n_samples = 1000

         sn = TypeIIbSNe()
         redshift = sn.sample_event_redshift(n_samples, rng=rng)
         params = sn.sed.sample_parameters(size=n_samples, rng=rng)
         params_grid = {pname: value[:, None] for pname, value in params.items()}

         # The all-sky rate, with no survey footprint applied.
         all_sky_rate = sn.all_sky_rate

         t_grid_rest = np.geomspace(0.1, 200, 300) * u.day
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
         ax.set_title("Peak-visible Type IIb rate")

         fig.tight_layout()

References
-----------

.. footbibliography::
