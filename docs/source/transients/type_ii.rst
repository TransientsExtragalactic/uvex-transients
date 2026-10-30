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
and SN 2024ggi. The ordinary IIP model is a cut-off power-law shock-cooling decline, a plateau and a
radioactive tail; the excess variant instead keeps an earlier two-exponential light curve with a logistic rise,
and a photospheric temperature with an additional early reheating bump. Both have a line-blanketed
blackbody photosphere, calibrated against Swift/UVOT UVW1 and UVW2 light curves. Type IIb events, which lost most but not all of their hydrogen, are modeled as
the sum of a shock cooling flash and a radioactively powered main peak. Each is implemented as its own transient population in a tab below.
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
           - :math:`z = 0.51`
           - --
           - Redshift beyond which, with 95% confidence, at most 1% of events drawn from the SED's
             priors are brighter than 24.5 AB (the UVEX 1 Dwell limit) in either UVEX band,
             rounded up. See the observability summary below.
         * - Duration
           - 100 days
           - --
           - Covers the plateau and the transition to the (unmodeled) nebular phase.

      .. _transients_type_ii_sed:

      .. rubric:: SED Model

      :class:`~uvex_transients.models.supernovae.IIp.TypeIIPSED` is a cut-off power-law shock-cooling
      decline, a hydrogen-recombination plateau and a radioactive tail, with a cooling, line-blanketed
      blackbody photosphere:

      .. math::

          L_\mathrm{bol}(t) = L_1\left(\frac{t}{1\,\mathrm{d}}\right)^{-\alpha} e^{-t/t_c}
          + \bigl[1-S_P(t)\bigr]L_p
          + S_P(t)\, L_\mathrm{Co}\, e^{-(t-t_P)/\tau_\mathrm{Co}},

      with the plateau-end switch

      .. math::

          S_P(t) = \frac{1}{1+e^{-(t-t_P)/\tau_\mathrm{drop}}}.

      The first term is the shock-cooling decline of the extended envelope: a power law in time, cut off
      exponentially once the emission comes from deeper in the envelope and hydrogen starts to recombine.
      The cutoff has to sit inside this term, because adding a constant plateau to a bare power law can only
      make the curve flatter. The second term is the hydrogen-recombination plateau at :math:`L_p`, and
      :math:`S_P` turns it off, and the radioactive tail on, at the plateau-end epoch :math:`t_P`. The tail is
      a pure :math:`^{56}\mathrm{Co}` exponential from :math:`L_\mathrm{Co}` at :math:`t_P`.

      There is no rise: the bolometric luminosity is largest at the earliest times, as it is for most of the
      SNe II in the calibration sample. The power law is evaluated no earlier than :math:`0.1\ \mathrm{d}`, and
      :math:`L_\mathrm{bol}` is zero before explosion. The planar shock-breakout phase,
      :math:`L\propto t^{-4/3}` for the first few hours, is not modeled.

      The photospheric temperature falls from a hot start to a cool floor as a logistic in :math:`\ln t`:

      .. math::

          T(t) = T_\mathrm{floor} + (T_0 - T_\mathrm{floor})\,\frac{1}{1 + (t/t_T)^{k_T}}.

      It sits near :math:`T_0` until a fraction of :math:`t_T`, is halfway to the floor at :math:`t_T`, and
      falls as :math:`t^{-k_T}` afterwards. :math:`T_0` is set by expectation rather than by data: the
      shock-heated photosphere is expected to start at :math:`\sim5\times10^4` K or hotter, and the prior is
      centered on the cooler end of that. The transition epoch (:math:`\sim6` d) and steepness are tied to the
      Swift/UVOT UVW1 light curves, which put the photosphere at :math:`\sim10^4` K by
      :math:`\sim10`-:math:`15` d.

      A plain blackbody does not describe the UV of a Type II SN well. Line blanketing makes the UV fainter
      than a blackbody at the optical temperature, and the photosphere here is therefore line-blanketed in the
      same way as the :ref:`Type Ia SED <transients_type_i>` (see its SED Model section for the form of the
      suppression): the UV blueward of :math:`\nu_\mathrm{break}` falls as :math:`\nu^{-\beta}`, and the shape
      is divided by its integral at each epoch, so that :math:`L_\mathrm{bol}(t)` is unchanged and the flux
      removed from the UV is returned redward of the break. ``break_frequency`` (:math:`0.75\times10^{15}`
      Hz) and ``beta`` (5) are fixed at the Type Ia values.

      The temperature is a compromise between the UV and the optical data, and not a fit to the optical
      temperatures. ``T_floor``, ``t_T`` and ``k_T`` were fit against the Swift/UVOT UVW1 light curves of
      the Type IIP SNe in ``test_data/transients`` (epochs from 2 to 50 d), with ``T0`` held at its prior and
      the optical temperatures of SN 1999em, SN 2003hn, SN 2012aw, SN 2012A and SN 2008in weighted down. The
      UVOT light curves are mostly timed from discovery rather than explosion, which the calibration treats
      as a random 0 to 5 d lag. That fit predates the line blanketing. After it was added, ``T_floor`` was
      raised by hand to :math:`10^{3.74}\ \mathrm{K}` (about 5500 K, close to the hydrogen-recombination
      temperature) and ``k_T`` lowered to between 1.5 and 2.0, giving a warmer, smoother cooling transition
      than the fit alone.

      .. dropdown:: Parameter priors

         The light-curve priors come from fitting the bolometric form above to each curve in the Type IIP
         calibration set (the curves of :footcite:t:`dallora2014`, :footcite:t:`faran2018`,
         :footcite:t:`bersten2009` and the 35 comparison SNe II of :footcite:t:`jacobsongalan2024`), keeping
         the 28 with a resolved plateau drop and a good fit, and then adjusting the prior widths until the
         prior-predictive spread of :math:`L_\mathrm{bol}` matched the observed spread at epochs from 2 to
         280 d. ``tau_drop`` is the median of the per-object fits. The fits prefer :math:`\alpha\simeq0` for
         most objects (a plain exponential decline with :math:`t_c\sim14` d), so ``alpha`` is only bounded to
         :math:`[0, 0.35]`. ``L_1`` is a truncated normal, limited to the range of the fitted values, which
         keeps the bright tail of the population within what is observed. The priors are independent,
         although in the data :math:`L_1` and :math:`L_p` are correlated, so the population is somewhat broader
         than the observed one at fixed epoch pairs. The temperature-law and line-blanketing priors are
         described above.

         .. list-table::
            :header-rows: 1
            :widths: 16 26 42 16

            * - Parameter
              - Symbol
              - Prior
              - Notes
            * - ``L_1``
              - :math:`L_1`
              - TruncatedNormal(:math:`\log_{10}(L_1/\mathrm{erg\,s^{-1}})`; mean=42.45, :math:`\sigma`\=1.1, bounds 41.3 to 43.5)
              - Shock-cooling luminosity scale: the power law's value at 1 d, before the cutoff.
            * - ``alpha``
              - :math:`\alpha`
              - Uniform(0, 0.35)
              - Shock-cooling power-law decay index.
            * - ``t_c``
              - :math:`t_c`
              - Uniform(9 d, 13 d)
              - Exponential cutoff time of the shock-cooling decline.
            * - ``L_p``
              - :math:`L_p`
              - Normal(:math:`\log_{10}(L_p/\mathrm{erg\,s^{-1}})`; mean=41.76, :math:`\sigma`\=0.35)
              - Plateau luminosity (median :math:`\sim6\times10^{41}` erg/s).
            * - ``t_P``
              - :math:`t_P`
              - Uniform(75 d, 135 d)
              - Plateau-end / radioactive-tail-onset epoch.
            * - ``tau_drop``
              - :math:`\tau_\mathrm{drop}`
              - Fixed (7 d)
              - Plateau-end transition width.
            * - ``L_Co``
              - :math:`L_\mathrm{Co}`
              - Normal(:math:`\log_{10}(L_\mathrm{Co}/\mathrm{erg\,s^{-1}})`; mean=41.03, :math:`\sigma`\=0.35)
              - Radioactive-tail luminosity at ``t_P``.
            * - ``tau_Co``
              - :math:`\tau_\mathrm{Co}`
              - Fixed (111.4 d)
              - Radioactive-tail decay timescale, fixed to the :math:`^{56}\mathrm{Co}` e-folding
                time.
            * - ``T0``
              - :math:`T_0`
              - Normal(:math:`\log_{10}(T_0/\mathrm{K})`; mean=4.7, :math:`\sigma`\=0.15)
              - Temperature at :math:`t=0` (median :math:`5\times10^4` K). Not constrained by data.
            * - ``T_floor``
              - :math:`T_\mathrm{floor}`
              - Normal(:math:`\log_{10}(T_\mathrm{floor}/\mathrm{K})`; mean=3.74, :math:`\sigma`\=0.04)
              - Late-time temperature (median :math:`5.5\times10^3` K). Raised by hand after the line blanketing was added.
            * - ``t_T``
              - :math:`t_T`
              - Normal(:math:`\log_{10}(t_T/\mathrm{d})`; mean=0.77, :math:`\sigma`\=0.2)
              - Epoch at which the photosphere is halfway from :math:`T_0` to :math:`T_\mathrm{floor}`
                (median :math:`\sim6` d). Fit to the Swift/UVOT UVW1 light curves.
            * - ``k_T``
              - :math:`k_T`
              - Uniform(1.5, 2.0)
              - Steepness of the cooling transition. Lowered by hand after the line blanketing was added.
            * - ``break_frequency``
              - :math:`\nu_\mathrm{break}`
              - Fixed (:math:`0.75\times10^{15}` Hz)
              - Frequency blueward of which line blanketing suppresses the flux. The Type Ia value.
            * - ``beta``
              - :math:`\beta`
              - Fixed (5)
              - Power-law index of the line-blanketing suppression. The Type Ia value.

      .. rubric:: Simulated Light Curves

      The plot below draws 300 random parameter realizations from the priors above and shows the
      resulting bolometric light curves and photospheric temperatures, overlaid with the observed
      light curves and temperatures of several Type IIP SNe :footcite:p:`dallora2014,faran2018`. The
      bolometric panel also shows, in grey, the light curves of the 35 "comparison" SNe II of
      :footcite:t:`jacobsongalan2024`, which show no early-time flash features. The temperature law was set
      by the UV data (see above), so the temperature panel is not a fit to the optical temperatures shown.

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

         comparison_sne = [e for e in archive.events("supernovae/IIP") if e.endswith("_jacobsongalan2024")]
         for i, suffix in enumerate(comparison_sne):
             lbol_obs = archive.table("supernovae/IIP", suffix, "L_bol")
             ax_L.plot(
                 lbol_obs["time"].to_value(u.day), lbol_obs["L_bol"].to_value(u.erg / u.s),
                 color="0.6", lw=0.6, marker=".", ms=3, alpha=0.6, zorder=3,
                 label=f"Jacobson-Galan+24 (n={len(comparison_sne)})" if i == 0 else None,
             )

         ax_L.set_xscale("log")
         ax_L.set_yscale("log")
         ax_L.set_ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
         ax_L.set_title("Type IIP SNe: simulated bolometric light curves (n=300)")
         ax_L.legend(loc="lower left", fontsize=8, frameon=False)
         ax_L.set_ylim([1e40,None])
         ax_L.set_xlim(0.3, 300)
         ax_T.set_yscale("log")
         ax_T.set_xlabel("Time since explosion [days]")
         ax_T.set_ylabel("Photospheric temperature [K]")
         ax_T.set_ylim([1e3,None])

         fig.tight_layout()

      .. rubric:: UV Calibration

      The plots below overlay the Swift/UVOT UVW1 (2600 Å) and UVW2 (1928 Å) light curves of
      Type IIP supernovae from the Swift Optical/Ultraviolet Supernova Archive (SOUSA) on 100 random draws from the priors above,
      integrated through the corresponding UVOT transmission curve from the SVO Filter Profile Service. Both are shown as
      :math:`\nu L_\nu` in the observed band with no K-correction. Archive supernovae are drawn at random, and the
      legend gives how many are shown. Only those with an estimated explosion date (black) are timed from explosion;
      the rest (grey) are timed from discovery, so they sit to the left of where the model places the same phase, by
      an amount that differs from event to event.

      The cooling-law parameters ``T_floor``, ``t_T`` and ``k_T`` were fit against the UVW1 light curves (see the SED
      Model above), and the line-blanketing parameters are those of Type Ia, unchanged. 11 Type IIP supernovae have at
      least four UVW1 detections, and 9 have at least four UVW2 detections.

      .. plot::
         :include-source: false

         import matplotlib.pyplot as plt

         from uvex_transients.models.supernovae import TypeIIPSED
         from uvex_transients.utils.plotting import plot_swift_uv_calibration

         plot_swift_uv_calibration(TypeIIPSED, "IIP", band="UVW1")
         plt.show()

      .. plot::
         :include-source: false

         import matplotlib.pyplot as plt

         from uvex_transients.models.supernovae import TypeIIPSED
         from uvex_transients.utils.plotting import plot_swift_uv_calibration

         plot_swift_uv_calibration(TypeIIPSED, "IIP", band="UVW2")
         plt.show()

      .. rubric:: Observability Summary

      The plots below show how the redshift limit follows from the SED. We draw 1000 sets of
      parameters from the priors above and evaluate each of them across a grid of redshifts, in the
      UVEX FUV and NUV bands, over the whole rest-frame light curve. The left panel shows each
      draw's peak apparent AB magnitude, in whichever band is brighter, against redshift: a draw is
      detectable out to the redshift where its line crosses the UVEX 1 Dwell limit of
      :math:`m<24.5`. The right panel shows the redshift beyond which, with 95% confidence, at most
      1% of the population is detectable, as a function of the magnitude limit. The redshift limit
      adopted above, :math:`z=0.51`, is its value at :math:`m=24.5`, rounded up and derived on a
      finer grid than this plot's.

      The limit bounds the fraction of the *population* that is detectable beyond it, not the
      fraction of detected events, and it does not include Milky Way extinction or sky position.

      .. plot::
         :include-source: false

         import numpy as np
         import matplotlib.pyplot as plt

         from uvex_transients.missions import uvex_fast as uvex
         from uvex_transients.transients.supernovae import TypeIIPSNe
         from uvex_transients.utils.plotting import plot_detection_horizon

         transient = TypeIIPSNe()
         curve, grid = transient.get_detection_horizon(
             np.arange(22.0, 27.01, 0.5),
             uvex.detector.bandpasses,
             z_min=transient.redshift_limit / 4,
             z_max=4 * transient.redshift_limit,
             n_z=45,
             rng=0,
             progress=False,
         )
         plot_detection_horizon(transient, curve, grid, mag_limit=24.5)
         plt.show()

      The anticipated rate detectable by UVEX at these limits is as follows, assuming that any
      event above the :math:`m<24.5` limit is detectable, and that the population is isotropic and
      homogeneous in comoving volume out to its redshift limit:

      .. plot::
         :include-source: false

          import numpy as np
          import matplotlib.pyplot as plt
          from astropy import units as u

          from uvex_transients.missions import uvex_fast as uvex
          from uvex_transients.transients.supernovae import TypeIIPSNe
          from uvex_transients.utils.plotting import add_funnel_legend, get_band_color, plot_rate_bars

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
          ax.set_title("Peak-visible Type IIP SNe rate")
          add_funnel_legend(ax, loc="lower right")

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
           - :math:`z = 0.53`
           - --
           - Redshift beyond which, with 95% confidence, at most 1% of events drawn from the SED's
             priors are brighter than 24.5 AB (the UVEX 1 Dwell limit) in either UVEX band,
             rounded up. See the observability summary below.
         * - Duration
           - 100 days
           - --
           - Same window as ordinary Type IIP.

      .. rubric:: SED Model

      :class:`~uvex_transients.models.supernovae.IIp.TypeIIPExcessSED` is a subclass of
      :class:`~uvex_transients.models.supernovae.IIp.TypeIIPSED` that keeps the earlier light curve, two
      shared-onset exponentials (an early cooling-phase decline onto a constant plateau) plus a switched
      radioactive tail,

      .. math::

          L_\mathrm{bol}(t) = S_0(t)\,\bigl[1-S_P(t)\bigr]
          \Bigl[L_\mathrm{pk}\,e^{-(t-t_0)/\tau_\mathrm{cool}} + L_p\Bigr]
          + S_P(t)\, L_\mathrm{Co}\, e^{-(t-t_P)/\tau_\mathrm{Co}},

      with logistic switches :math:`S_0(t) = [1+e^{-(t-t_0)/\tau_\mathrm{rise}}]^{-1}` and
      :math:`S_P(t) = [1+e^{-(t-t_P)/\tau_\mathrm{drop}}]^{-1}`. The logistic rise at :math:`t_0` and the bright
      early cooling-phase peak :math:`L_\mathrm{pk}` describe the brighter, faster early phase seen in
      SN 2023ixf and SN 2024ggi, rather than the smooth, monotonic early decline of ordinary Type IIP SNe.

      The temperature law also differs. It is a single-power-law cooling law, describing a
      hot initial flash (:math:`T_0 \sim 10^5` K) that has cooled by :math:`\sim1` d, plus a reheating
      bump that reproduces the rise of the photospheric temperature to :math:`\sim2\times10^4` K at
      :math:`\sim3` d seen in both objects:

      .. math::

          T(t) = T_\mathrm{floor} + (T_0 - T_\mathrm{floor})\left(1 + \frac{t}{\tau_T}\right)^{-\alpha_T}
          + \Delta T_\mathrm{bump}\,\frac{g(t/\tau_b)}{g(\beta_\mathrm{bump}/\alpha_\mathrm{bump})},
          \qquad g(x) = \frac{x^{\beta_\mathrm{bump}}}{(1 + x)^{\alpha_\mathrm{bump} + \beta_\mathrm{bump}}},

      with :math:`\tau_b = t_\mathrm{bump}\,\alpha_\mathrm{bump}/\beta_\mathrm{bump}`, so that the
      bump rises as :math:`t^{\beta_\mathrm{bump}}`, peaks at exactly :math:`t_\mathrm{bump}` with
      height :math:`\Delta T_\mathrm{bump}`, and decays as :math:`t^{-\alpha_\mathrm{bump}}`. The
      temperature is therefore not monotonic: it cools from the flash to a trough of
      :math:`\sim1.3\times10^4` K at :math:`\sim1` d, rises to the bump, and cools again.

      The blackbody is line-blanketed as for ordinary Type IIP (see the Type IIP tab and the
      :ref:`Type Ia SED <transients_type_i>`), with the same ``break_frequency`` of
      :math:`0.75\times10^{15}` Hz but a weaker suppression, ``beta`` = 3.5 instead of 5, tuned against the
      Swift/UVOT light curves of SN 2023ixf and SN 2024ggi. The shape is divided by its integral, so
      :math:`L_\mathrm{bol}(t)` is unchanged.

      .. dropdown:: Parameter priors

         Hand-tuned against SN 2023ixf :footcite:p:`hsu2025` and SN 2024ggi
         :footcite:p:`chen2024`, the only two objects of this kind currently in
         ``test_data/transients``. There is no temperature data before
         :math:`\sim0.8` d, so the flash parameters (``T0``, ``tau_T``, ``alpha_T``) are set by
         expectation. ``T_floor`` and the bump parameters come from fits to the two objects
         (with ``beta_bump`` only bounded below, :math:`\gtrsim5`).

         .. list-table::
            :header-rows: 1
            :widths: 16 26 42 16

            * - Parameter
              - Symbol
              - Prior
              - Notes
            * - ``t0``
              - :math:`t_0`
              - Uniform(0.5 d, 3.5 d)
              - Explosion/rise reference epoch.
            * - ``tau_rise``
              - :math:`\tau_\mathrm{rise}`
              - Normal(:math:`\log_{10}(\tau_\mathrm{rise}/\mathrm{d})`; mean=-0.2, :math:`\sigma`\=0.15)
              - Rise timescale (median :math:`\sim0.63` d).
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
            * - ``tau_drop``
              - :math:`\tau_\mathrm{drop}`
              - Fixed (10.5 d)
              - Plateau-end transition width.
            * - ``L_Co``
              - :math:`L_\mathrm{Co}`
              - Normal(:math:`\log_{10}(L_\mathrm{Co}/\mathrm{erg\,s^{-1}})`; mean=41.5, :math:`\sigma`\=0.3)
              - Radioactive-tail luminosity at ``t_P``.
            * - ``tau_Co``
              - :math:`\tau_\mathrm{Co}`
              - Fixed (111.4 d)
              - Radioactive-tail decay timescale, fixed to the :math:`^{56}\mathrm{Co}` e-folding time.
            * - ``T0``
              - :math:`T_0`
              - Normal(:math:`\log_{10}(T_0/\mathrm{K})`; mean=5.0, :math:`\sigma`\=0.1)
              - Temperature at :math:`t=0` (median :math:`10^5` K), the hot flash. Not constrained by data.
            * - ``T_floor``
              - :math:`T_\mathrm{floor}`
              - Normal(:math:`\log_{10}(T_\mathrm{floor}/\mathrm{K})`; mean=3.69, :math:`\sigma`\=0.06)
              - Late-time temperature (median :math:`4.9\times10^3` K).
            * - ``tau_T``
              - :math:`\tau_T`
              - Uniform(0.15 d, 0.3 d)
              - Cooling timescale of the initial flash.
            * - ``alpha_T``
              - :math:`\alpha_T`
              - Uniform(1.8, 2.4)
              - Cooling power-law index of the initial flash.
            * - ``T_bump``
              - :math:`\Delta T_\mathrm{bump}`
              - Normal(:math:`\log_{10}(\Delta T_\mathrm{bump}/\mathrm{K})`; mean=4.19, :math:`\sigma`\=0.08)
              - Peak temperature excess of the reheating bump (median :math:`1.55\times10^4` K).
            * - ``t_bump``
              - :math:`t_\mathrm{bump}`
              - Uniform(2.4 d, 3.6 d)
              - Epoch of the bump's peak.
            * - ``beta_bump``
              - :math:`\beta_\mathrm{bump}`
              - Uniform(5, 12)
              - Rise power-law index of the bump.
            * - ``alpha_bump``
              - :math:`\alpha_\mathrm{bump}`
              - Uniform(1.6, 2.4)
              - Decay power-law index of the bump.
            * - ``break_frequency``
              - :math:`\nu_\mathrm{break}`
              - Fixed (:math:`0.75\times10^{15}` Hz)
              - Frequency blueward of which line blanketing suppresses the flux. The Type Ia value.
            * - ``beta``
              - :math:`\beta`
              - Fixed (3.5)
              - Power-law index of the line-blanketing suppression. Weaker than the ordinary Type IIP value of 5.

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

      .. rubric:: UV Calibration

      The plots below overlay the Swift/UVOT UVW1 (2600 Å) and UVW2 (1928 Å) light curves of
      Type IIP + Excess supernovae (SN 2023ixf and SN 2024ggi) from the Swift Optical/Ultraviolet Supernova Archive (SOUSA) on 100 random draws from the priors above,
      integrated through the corresponding UVOT transmission curve from the SVO Filter Profile Service. Both are shown as
      :math:`\nu L_\nu` in the observed band with no K-correction. Archive supernovae are drawn at random, and the
      legend gives how many are shown. Only those with an estimated explosion date (black) are timed from explosion;
      the rest (grey) are timed from discovery, so they sit to the left of where the model places the same phase, by
      an amount that differs from event to event.

      These two supernovae are the whole Swift sample for this population, and the whole sample the SED was tuned
      against, so the comparison is not independent of the priors.

      .. plot::
         :include-source: false

         import matplotlib.pyplot as plt

         from uvex_transients.models.supernovae import TypeIIPExcessSED
         from uvex_transients.utils.plotting import plot_swift_uv_calibration

         plot_swift_uv_calibration(TypeIIPExcessSED, "IIP+Excess", band="UVW1")
         plt.show()

      .. plot::
         :include-source: false

         import matplotlib.pyplot as plt

         from uvex_transients.models.supernovae import TypeIIPExcessSED
         from uvex_transients.utils.plotting import plot_swift_uv_calibration

         plot_swift_uv_calibration(TypeIIPExcessSED, "IIP+Excess", band="UVW2")
         plt.show()

      .. rubric:: Observability Summary

      The plots below show how the redshift limit follows from the SED. We draw 1000 sets of
      parameters from the priors above and evaluate each of them across a grid of redshifts, in the
      UVEX FUV and NUV bands, over the whole rest-frame light curve. The left panel shows each
      draw's peak apparent AB magnitude, in whichever band is brighter, against redshift: a draw is
      detectable out to the redshift where its line crosses the UVEX 1 Dwell limit of
      :math:`m<24.5`. The right panel shows the redshift beyond which, with 95% confidence, at most
      1% of the population is detectable, as a function of the magnitude limit. The redshift limit
      adopted above, :math:`z=0.53`, is its value at :math:`m=24.5`, rounded up and derived on a
      finer grid than this plot's.

      The limit bounds the fraction of the *population* that is detectable beyond it, not the
      fraction of detected events, and it does not include Milky Way extinction or sky position.

      .. plot::
         :include-source: false

         import numpy as np
         import matplotlib.pyplot as plt

         from uvex_transients.missions import uvex_fast as uvex
         from uvex_transients.transients.supernovae import TypeIIPExcessSNe
         from uvex_transients.utils.plotting import plot_detection_horizon

         transient = TypeIIPExcessSNe()
         curve, grid = transient.get_detection_horizon(
             np.arange(22.0, 27.01, 0.5),
             uvex.detector.bandpasses,
             z_min=transient.redshift_limit / 4,
             z_max=4 * transient.redshift_limit,
             n_z=45,
             rng=0,
             progress=False,
         )
         plot_detection_horizon(transient, curve, grid, mag_limit=24.5)
         plt.show()

      The anticipated rate detectable by UVEX at these limits is as follows, assuming that any
      event above the :math:`m<24.5` limit is detectable, and that the population is isotropic and
      homogeneous in comoving volume out to its redshift limit:

      .. plot::
         :include-source: false

          import numpy as np
          import matplotlib.pyplot as plt
          from astropy import units as u

          from uvex_transients.missions import uvex_fast as uvex
          from uvex_transients.transients.supernovae import TypeIIPExcessSNe
          from uvex_transients.utils.plotting import add_funnel_legend, get_band_color, plot_rate_bars

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
          ax.set_title("Peak-visible Type IIP + Excess SNe rate")
          add_funnel_legend(ax, loc="lower right")

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
      component, cut at the edge of its regime of validity, this population adds the radioactively powered
      main peak and evaluates the shock cooling at every epoch, so a single SED spans the whole light curve.

      This population is implemented by
      :class:`~uvex_transients.transients.supernovae.TypeIIbSNe`, pairing
      :class:`~uvex_transients.models.supernovae.IIb.TypeIIbSED` with the rate/duration metadata
      described below.

      .. note::

         The shock cooling priors are those of :class:`~uvex_transients.models.shock_cooling.MoragShockCoolingSED`
         and the nickel priors are those of the Type Ib SED
         (:class:`~uvex_transients.models.supernovae.Ibc.TypeIbSED`). Neither is a fit to Type IIb events, and
         the duration below was derived for an earlier, phenomenological Type IIb SED. It needs to be
         recomputed.

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
           - :math:`z = 1.0`
           - --
           - Redshift beyond which, with 95% confidence, at most 1% of events drawn from the SED's
             priors are brighter than 24.5 AB (the UVEX 1 Dwell limit) in either UVEX band,
             rounded up. Only epochs after 0.1 d are considered, because the shock cooling
             luminosity grows without bound toward :math:`t\to0`. See the observability summary below.
         * - Duration
           - 200 days
           - --
           - Long enough to cover the shock-cooling peak (where present), the dip, the radioactive main
             peak, and its subsequent decline -- unlike
             :class:`~uvex_transients.transients.supernovae.ShockCoolingIIb`'s much shorter 5 day
             window, which covers only the first of those phases.

      .. rubric:: SED Model

      :class:`~uvex_transients.models.supernovae.IIb.TypeIIbSED` adds two emission components,

      .. math::

          L_\nu(\nu, t) = L_\nu^\mathrm{shock}(\nu, t) + L_\nu^\mathrm{Ni}(\nu, t).

      The first is the full shock cooling SED of :footcite:t:`2024MNRAS.528.7137M`
      (:class:`~uvex_transients.models.shock_cooling.MoragShockCoolingSED`), which dominates the first
      days. It is evaluated at every epoch, without the cut at the edges of that paper's regime of
      validity, so it does not switch off after a day or two and the luminosity is not zero through the
      trough. The price is that the formulas are not valid before :math:`t_\mathrm{min}`, where the
      luminosity grows without bound toward :math:`t \to 0`. The second is the line-blanketed Arnett
      nickel light curve of :class:`~uvex_transients.models.supernovae.Ibc.TypeIbSED`, which produces the
      main peak at about 20 days. Together they give a luminous decline after breakout, a trough, and a
      rise to the nickel peak. The two components are independent, so nothing ties the envelope mass of the
      shock cooling to the ejecta mass of the nickel component.

      .. dropdown:: Parameter priors

         The shock cooling parameters have the priors of
         :class:`~uvex_transients.models.shock_cooling.MoragShockCoolingSED`, and the nickel and
         line-blanketing parameters those of :class:`~uvex_transients.models.supernovae.Ibc.TypeIbSED`, except
         ``M_Ni`` and ``beta``. The line blanketing applies to the nickel component only: the shock cooling
         component is an unblanketed blackbody.

         .. list-table::
            :header-rows: 1
            :widths: 16 12 38 34

            * - Parameter
              - Symbol
              - Prior
              - Notes
            * - ``v_star``
              - :math:`v_*`
              - LogNormal(:math:`\sigma`\=0.35) around :math:`6.3\times10^8` cm/s
              - Scale velocity of the shock near the stellar surface.
            * - ``radius``
              - :math:`R`
              - LogNormal(:math:`\sigma`\=0.85) around :math:`6\times10^{12}` cm
              - Progenitor stellar radius.
            * - ``opacity``
              - :math:`\kappa`
              - Fixed (0.34 cm\ :sup:`2` g\ :sup:`-1`)
              - Electron-scattering opacity of the shock cooling.
            * - ``envelope_mass``
              - :math:`M_E`
              - LogNormal(:math:`\sigma`\=1.0) around :math:`0.05\ M_\odot`
              - Hydrogen-rich envelope mass.
            * - ``core_mass``
              - :math:`M_C`
              - LogNormal(:math:`\sigma`\=0.35) around :math:`2.8\ M_\odot`
              - Core mass.
            * - ``M_Ni``
              - :math:`M_\mathrm{Ni}`
              - TruncatedNormal(mean=0.1, :math:`\sigma`\=0.07, bounds 0.01 to 0.4 :math:`M_\odot`)
              - Nickel-56 mass. Lower than the Type Ib value.
            * - ``M_ej``
              - :math:`M_\mathrm{ej}`
              - TruncatedNormal(mean=2.6, :math:`\sigma`\=1.1, bounds 0.1 to 8 :math:`M_\odot`)
              - Ejecta mass.
            * - ``v_ej``
              - :math:`v_\mathrm{ej}`
              - TruncatedNormal(mean=9.9, :math:`\sigma`\=1.4, bounds 4 to 16; units of :math:`10^3` km/s)
              - Constant ejecta velocity.
            * - ``kappa``
              - :math:`\kappa`
              - Fixed (0.06 cm\ :sup:`2` g\ :sup:`-1`)
              - Grey optical opacity of the ejecta.
            * - ``kappa_gamma``
              - :math:`\kappa_\gamma`
              - Fixed (0.04 cm\ :sup:`2` g\ :sup:`-1`)
              - Opacity to high-energy photons.
            * - ``T_floor``
              - :math:`T_\mathrm{floor}`
              - TruncatedNormal(mean=5000 K, :math:`\sigma`\=1000 K, bounds 3000 K to 10000 K)
              - Minimum photospheric temperature of the nickel component.
            * - ``break_frequency``
              - :math:`\nu_\mathrm{break}`
              - Fixed (:math:`0.75\times10^{15}` Hz)
              - Frequency blueward of which line blanketing suppresses the nickel component.
            * - ``beta``
              - :math:`\beta`
              - Fixed (4.0)
              - Power-law index of the line-blanketing suppression. Weaker than the Type Ib value of 5.

      .. rubric:: Simulated Light Curves

      The plot below draws 300 random parameter realizations from the priors above and shows the
      resulting bolometric light curves and photospheric temperatures, overlaid with the observed
      light curves and temperatures of several Type IIb SNe
      :footcite:p:`richmond1994,taubenberger2011,moralesgaroffolo2014,moralesgaroffolo2015,yamanaka2025,prentice2019`.
      The spread shows the early shock cooling decline, the trough and the nickel peak, with the depth of
      the trough set by the shock cooling parameters. The shock cooling is evaluated outside its regime of
      validity at the earliest times, so the very early luminosity of the brightest draws should not be
      read as a prediction.

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



      .. rubric:: UV Calibration

      The plots below overlay the Swift/UVOT UVW1 (2600 Å) and UVW2 (1928 Å) light curves of
      Type IIb supernovae from the Swift Optical/Ultraviolet Supernova Archive (SOUSA) on 100 random draws from the priors above,
      integrated through the corresponding UVOT transmission curve from the SVO Filter Profile Service. Both are shown as
      :math:`\nu L_\nu` in the observed band with no K-correction. Archive supernovae are drawn at random, and the
      legend gives how many are shown. Only those with an estimated explosion date (black) are timed from explosion;
      the rest (grey) are timed from discovery, so they sit to the left of where the model places the same phase, by
      an amount that differs from event to event.

      Only the nickel component of this SED is line-blanketed (``beta`` = 4), so the UV in the first days is
      set by the unblanketed shock cooling component. 7 Type IIb supernovae have at least four UVW1 detections, and
      6 have at least four UVW2 detections.

      .. plot::
         :include-source: false

         import matplotlib.pyplot as plt

         from uvex_transients.models.supernovae import TypeIIbSED
         from uvex_transients.utils.plotting import plot_swift_uv_calibration

         plot_swift_uv_calibration(TypeIIbSED, "IIb", band="UVW1")
         plt.show()

      .. plot::
         :include-source: false

         import matplotlib.pyplot as plt

         from uvex_transients.models.supernovae import TypeIIbSED
         from uvex_transients.utils.plotting import plot_swift_uv_calibration

         plot_swift_uv_calibration(TypeIIbSED, "IIb", band="UVW2")
         plt.show()

      .. rubric:: Observability Summary

      The plots below show how the redshift limit follows from the SED. We draw 1000 sets of
      parameters from the priors above and evaluate each of them across a grid of redshifts, in the
      UVEX FUV and NUV bands, over the whole rest-frame light curve. The left panel shows each
      draw's peak apparent AB magnitude, in whichever band is brighter, against redshift: a draw is
      detectable out to the redshift where its line crosses the UVEX 1 Dwell limit of
      :math:`m<24.5`. The right panel shows the redshift beyond which, with 95% confidence, at most
      1% of the population is detectable, as a function of the magnitude limit. The redshift limit
      adopted above, :math:`z=1.0`, is its value at :math:`m=24.5`, rounded up and derived on a
      finer grid than this plot's. Epochs earlier than 0.1 d are excluded from the light curve, because the
      shock cooling luminosity of this SED grows without bound toward :math:`t\to0`.

      The limit bounds the fraction of the *population* that is detectable beyond it, not the
      fraction of detected events, and it does not include Milky Way extinction or sky position.

      .. plot::
         :include-source: false

         import numpy as np
         import matplotlib.pyplot as plt
         from astropy import units as u

         from uvex_transients.missions import uvex_fast as uvex
         from uvex_transients.transients.supernovae import TypeIIbSNe
         from uvex_transients.utils.plotting import plot_detection_horizon

         transient = TypeIIbSNe()
         curve, grid = transient.get_detection_horizon(
             np.arange(22.0, 27.01, 0.5),
             uvex.detector.bandpasses,
             t_min=0.1 * u.day,
             z_min=transient.redshift_limit / 4,
             z_max=4 * transient.redshift_limit,
             n_z=45,
             rng=0,
             progress=False,
         )
         plot_detection_horizon(transient, curve, grid, mag_limit=24.5)
         plt.show()

      The anticipated rate detectable by UVEX at this limit is as follows, assuming that any event above
      the :math:`m<24.5` limit is detectable, and that the population is isotropic and homogeneous in
      comoving volume out to its redshift limit:

      .. plot::
         :include-source: false

         import numpy as np
         import matplotlib.pyplot as plt
         from astropy import units as u

         from uvex_transients.missions import uvex_fast as uvex
         from uvex_transients.transients.supernovae import TypeIIbSNe
         from uvex_transients.utils.plotting import add_funnel_legend, get_band_color, plot_rate_bars

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
         ax.set_title("Peak-visible Type IIb rate")
         add_funnel_legend(ax, loc="lower right")

         fig.tight_layout()

References
-----------

.. footbibliography::
