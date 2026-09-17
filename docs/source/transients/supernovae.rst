.. _transients_supernovae:

Type IIP Supernovae
=====================

Type IIP core-collapse supernovae (CCSNe) are the explosions of hydrogen-rich massive stars: an
early cooling-phase decline (from the initial shock breakout) settles onto a
weeks-to-months-long luminosity "plateau" as a recombination front recedes through the ejecta,
followed by a radioactive tail powered by :math:`^{56}\mathrm{Co}` decay. A subset show a brighter,
hotter, faster early excess on top of this, attributed to shock breakout through and/or collisional
heating of close circumstellar material -- IXF/GGI-like objects, after the prototypes SN 2023ixf
and SN 2024ggi. Both variants share the same SED functional form and differ only in their parameter
priors, implemented as two sibling transient populations below.

.. tab-set::

   .. tab-item:: Type IIP

      Implemented by :class:`~uvex_transients.transients.supernovae.TypeIIPSNe`, pairing
      :class:`~uvex_transients.models.supernovae.IIp.TypeIIPSED` with the rate/duration metadata
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
           - :math:`R_\mathrm{CC}(z) = k h^2 \psi_\mathrm{UV}(z)`; Type IIP 40% of :math:`R_\mathrm{CC}(z)`
           - :footcite:t:`strolger2015`, :footcite:t:`madau2014`, :footcite:t:`li2011`
           - Tracks the cosmic star-formation history; 40% is the local Type IIP fraction of
             core-collapse SNe :footcite:p:`li2011`.
         * - Redshift limit
           - :math:`z = 0.8`
           - --
           - Matched to ordinary Type IIP peak luminosities.
         * - Duration
           - 100 days
           - --
           - Covers the plateau and the transition to the (unmodeled) nebular phase.

      .. _transients_supernovae_sed:

      SED Model
      ----------

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

      Simulated Light Curves
      ~~~~~~~~~~~~~~~~~~~~~~~

      The plot below draws 300 random parameter realizations from the priors above and shows the
      resulting bolometric light curves and photospheric temperatures, overlaid with the observed
      light curves and temperatures of several Type IIP SNe :footcite:p:`dallora2014,faran2018`.

      .. plot::
         :include-source: false

         from pathlib import Path

         import numpy as np
         import matplotlib.pyplot as plt
         from astropy import units as u
         from astropy.table import Table

         import uvex_transients
         from uvex_transients.models.supernovae import TypeIIPSED

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

         data_dir = Path(uvex_transients.__file__).parent.parent / "test_data" / "transients"
         observed_iip_sne = [
             ("1999em_bersten2009.txt", "SN 1999em", "o", "k"),
             ("2003hn_bersten2009.txt", "SN 2003hn", "s", "firebrick"),
             ("2012aw_dallora14.txt", "SN 2012aw", "^", "C2"),
             ("2012A_faran18.txt", "SN 2012A", "X", "C7"),
             ("2008in_faran18.txt", "SN 2008in", "*", "C8"),
         ]
         # Light-curve-only comparison objects (no photospheric temperature sequence available).
         lbol_only_iip_sne = [
             ("2004et_dallora14.txt", "SN 2004et", "v", "C4"),
             ("1992H_dallora14.txt", "SN 1992H", "D", "C5"),
             ("2009bw_dallora14.txt", "SN 2009bw", "P", "C6"),
         ]

         for suffix, label, marker, color in observed_iip_sne:
             lbol_obs = Table.read(data_dir / f"lbol_{suffix}", format="ascii")
             Tphot_obs = Table.read(data_dir / f"Tphot_{suffix}", format="ascii")
             ax_L.scatter(
                 lbol_obs["time"], lbol_obs["L_bol"],
                 marker=marker, s=28, color=color, edgecolor="white", linewidth=0.5, zorder=5,
                 label=label,
             )
             ax_T.scatter(
                 Tphot_obs["time"], Tphot_obs["T_phot"],
                 marker=marker, s=28, color=color, edgecolor="white", linewidth=0.5, zorder=5,
                 label=label,
             )

         for suffix, label, marker, color in lbol_only_iip_sne:
             lbol_obs = Table.read(data_dir / f"lbol_{suffix}", format="ascii")
             ax_L.scatter(
                 lbol_obs["time"], lbol_obs["L_bol"],
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

      Observability Summary
      ----------------------

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

          # Convert the integrated rate per steradian to an all-sky rate.
          all_sky_rate = 4 * np.pi * sn.integrated_event_rate * u.sr

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
           - 30% of the Type IIP rate
           - :footcite:t:`bruch2023`
           - Reflects the high incidence of early CSM-interaction signatures found among Type II
             SNe (ZTF).
         * - Redshift limit
           - :math:`z = 2`
           - --
           - Higher than ordinary Type IIP, to admit this population's brighter, more UV-luminous
             early-cooling realizations.
         * - Duration
           - 100 days
           - --
           - Same window as ordinary Type IIP.

      SED Model
      ----------

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

      Simulated Light Curves
      ~~~~~~~~~~~~~~~~~~~~~~~

      The plot below draws 300 random parameter realizations from the priors above and shows the
      resulting bolometric light curves and photospheric temperatures, overlaid with the observed
      light curves and temperatures of SN 2023ixf :footcite:p:`hsu2025` and SN 2024ggi
      :footcite:p:`chen2024`.

      .. plot::
         :include-source: false

         from pathlib import Path

         import numpy as np
         import matplotlib.pyplot as plt
         from astropy import units as u
         from astropy.table import Table

         import uvex_transients
         from uvex_transients.models.supernovae import TypeIIPExcessSED

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

         data_dir = Path(uvex_transients.__file__).parent.parent / "test_data" / "transients"
         observed_excess_sne = [
             ("2023ixf_hsu2025.txt", "SN 2023ixf", "o", "k"),
             ("2024ggi_chen2024.txt", "SN 2024ggi", "s", "firebrick"),
         ]

         for suffix, label, marker, color in observed_excess_sne:
             lbol_obs = Table.read(data_dir / f"lbol_{suffix}", format="ascii")
             Tphot_obs = Table.read(data_dir / f"Tphot_{suffix}", format="ascii")
             ax_L.scatter(
                 lbol_obs["time"], lbol_obs["L_bol"],
                 marker=marker, s=28, color=color, edgecolor="white", linewidth=0.5, zorder=5,
                 label=label,
             )
             ax_T.scatter(
                 Tphot_obs["time"], Tphot_obs["T_phot"],
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

      Observability Summary
      ----------------------

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

          # Convert the integrated rate per steradian to an all-sky rate.
          all_sky_rate = 4 * np.pi * sn.integrated_event_rate * u.sr

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

References
-----------

.. footbibliography::
