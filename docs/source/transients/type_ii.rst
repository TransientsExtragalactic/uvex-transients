.. _transients_type_ii:

Type II Supernovae
====================

Type II core-collapse supernovae (CCSNe) are the explosions of massive stars that retained at
least part of their hydrogen envelope. Type IIP events are the explosions of hydrogen-rich stars: an
early cooling-phase decline (from the initial shock breakout) settles onto a
weeks-to-months-long luminosity "plateau" as a recombination front recedes through the ejecta,
followed by a radioactive tail powered by :math:`^{56}\mathrm{Co}` decay. A subset show a brighter,
hotter, faster early excess on top of this, attributed to shock breakout through and/or collisional
heating of close circumstellar material, like the prototypes SN 2023ixf and SN 2024ggi. The Type IIP model is a
plateau and radioactive tail with a cooling blackbody photosphere whose UV is suppressed by line blanketing,
calibrated against the ASAS-SN luminosity function, Swift/UVOT light curves and optical bolometric light curves and
temperatures. The events with an early excess from confined circumstellar material (CSM) are 36% of the same
population, drawn from it with a switch and a few parameters that apply to them only. Type IIb events, which lost most but not
all of their hydrogen, are modeled as the sum of a shock cooling flash and a radioactively powered main peak. Each is
implemented as its own transient population in a tab below.
(Hydrogen-free Type Ib and Ic supernovae are covered on the :ref:`Type I page <transients_type_i>`.)

.. tab-set::

   .. tab-item:: Type IIP

      Implemented by :class:`~uvex_transients.transients.supernovae.TypeIIPSNe`, pairing
      :class:`~uvex_transients.models.supernovae.IIp.TypeIIPSED` with the rate/duration metadata
      described below. A fraction of the population, 36% by default, has a brighter early excess from a confined shell
      of CSM, like SN 2023ixf and SN 2024ggi (IXF/GGI-like objects). The same model describes them, with a switch
      (the parameter ``csm``) and a few parameters that apply to those events only (see the SED Model below), so that
      they are not a separate population.

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
             :ref:`user_guide_transients_rate_uncertainty`. This is the rate of all Type IIP, including the
             events with an early excess, which are a fraction of it (``csm``, 36%, after the high incidence of
             early interaction signatures among Type II SNe found by :footcite:t:`bruch2023` and
             :footcite:t:`hinds2025`). The fraction has no uncertainty included in the rate.
         * - Redshift limit
           - :math:`z = 1.07`
           - --
           - Redshift beyond which, with 95% confidence, at most 1% of events drawn from the SED's
             priors are brighter than 24.5 AB (the UVEX 1 Dwell limit) in either UVEX band,
             rounded up. It comes from the hot early UV, which is an extrapolation at this redshift: see
             the observability summary below.
         * - Duration
           - 100 days
           - --
           - Covers the plateau and the transition to the (unmodeled) nebular phase.

      .. _transients_type_ii_sed:

      .. rubric:: SED Model

      :class:`~uvex_transients.models.supernovae.IIp.TypeIIPSED` is a closed-form model: an envelope with a plateau, a
      radioactive tail and, for the events with circumstellar material (CSM), a CSM pulse, with a colour temperature
      derived from the luminosity and a blackbody spectrum whose UV is suppressed by line blanketing. There is no ODE
      integration, and redshift, distance and dust are not part of it. The form follows the temperature and
      bolometric evolution of SNe II with a recombination floor :footcite:p:`faran2017`, the colour temperature of
      shock cooling and the departures from a blackbody that suppress the UV :footcite:p:`morag2023,morag2024`, and
      the interaction power that emerges in the UV and shortens the optical rise of events with CSM
      :footcite:p:`dessart2022,dessart2026`.

      **Bolometric luminosity.** The sum of an envelope, a radioactive tail and a CSM pulse,

      .. math::

          L_\mathrm{bol}(t) = L_\mathrm{env}(t) + L_\mathrm{Ni}(t) + L_\mathrm{csm}(t),

      with the plateau cutoff :math:`F(t)=1/\{1+\exp[(t-t_p)/w]\}`. The envelope is the shock-cooling excess and the
      plateau,

      .. math::

          L_\mathrm{env}(t) = L_{50}\,10^{-0.4\,s\,(t-50)/100}\,\bigl[1+a_e e^{-t/\tau_e}\bigr]
          \bigl[1-e^{-(t/t_r)^2}\bigr]F(t),

      with :math:`t` in days, so that :math:`L_{50}` is the plateau luminosity at day 50. The tail is gated to emerge
      only after the plateau, which is what keeps that true,

      .. math::

          L_\mathrm{Ni}(t) = M_\mathrm{Ni}\bigl[(\epsilon_\mathrm{Ni}-\epsilon_\mathrm{Co})e^{-t/\tau_\mathrm{Ni}}
          + \epsilon_\mathrm{Co}e^{-t/\tau_\mathrm{Co}}\bigr]\bigl[1-e^{-(T_0/t)^2}\bigr]\bigl[1-F(t)\bigr],

      with the specific heating rates :math:`\epsilon_\mathrm{Ni}=3.9\times10^{10}` and
      :math:`\epsilon_\mathrm{Co}=6.78\times10^{9}` erg/s/g and the decay times :math:`\tau_\mathrm{Ni}=8.8` and
      :math:`\tau_\mathrm{Co}=111.3` d. The CSM pulse, with :math:`x=t/t_c`, is :math:`L_c\,x^k e^{k(1-x)}` up to its
      peak at :math:`t_c` and :math:`L_c\exp[1-\sqrt{1+((t-t_c)/\tau_d)^2}]` after it. It equals :math:`L_c` at
      :math:`t_c`, with zero slope on both sides, rises as :math:`t^k` and decays exponentially with the time
      :math:`\tau_d`. :math:`L_c=0` is a plain IIP, and the bolometric luminosity is zero at explosion.

      **Colour temperature.** An expanding photosphere with a recombination floor. The temperature is derived from
      the luminosity, never specified:

      .. math::

          R(t) = R_0 + v_\mathrm{ph}t,\quad
          T_\mathrm{exp} = \left[\frac{L_\mathrm{bol}}{4\pi\sigma R^2}\right]^{1/4},\quad
          T_\mathrm{col} = \bigl[(f_cT_\mathrm{exp})^n + T_\mathrm{rec}^n\bigr]^{1/n}.

      :math:`R_0` is the progenitor radius of a plain IIP and the breakout radius when there is CSM.

      **Spectrum.** A blackbody at :math:`T_\mathrm{col}`, suppressed in the UV by line blanketing and renormalized
      to :math:`L_\mathrm{bol}`,

      .. math::

          L_\nu = L_\mathrm{bol}\,\frac{\pi B_\nu(T_\mathrm{col})\,S(\lambda,T_\mathrm{col})}{\sigma T_\mathrm{col}^4\,
          N(T_\mathrm{col})},\qquad
          S=\begin{cases}(\lambda/\lambda_c)^{\beta(T)}&\lambda<\lambda_c\\1&\lambda\ge\lambda_c\end{cases},\qquad
          \beta(T)=\frac{\beta_\mathrm{max}}{1+e^{(T-T_b)/\Delta T}},

      where :math:`N(T)=\int\pi B_\nu S\,d\nu/\sigma T^4` is the fraction of the blackbody left after the suppression.
      The spectral shape therefore integrates to 1 at every time: the flux removed from the UV is moved to longer
      wavelengths, not lost, so that :math:`L_\mathrm{bol}` is the luminosity actually emitted. The blanketing is off
      above :math:`\sim10^4` K, where :math:`N=1`, so the early, hot photosphere is a plain blackbody, and the
      renormalization only matters once the photosphere has cooled (:math:`N=0.94` at 6000 K). :math:`N(T)` is
      tabulated once for each set of the four UV parameters and interpolated linearly in :math:`\ln T`. There is no
      cutoff at the hydrogen Lyman limit.

      **Population.** The parameters are drawn from a hierarchical population, not independently. An event is a CSM
      event with probability :math:`f_\mathrm{csm}=0.36` (the parameter ``csm``), then :math:`L_{50}` (and, for a CSM
      event, :math:`L_c`) is drawn, then every other parameter given them.

      * **Luminosity trends.** The priors of ``s``, ``M_Ni`` and ``v_ph`` are those of an event with
        :math:`\log_{10}L_{50}=42`, and that of ``t_c`` of one with :math:`\log_{10}L_c=43.15`. The mean of each, in
        the variable the parameter is sampled in, moves by a slope times the distance of the event's luminosity from
        that reference, with its truncation bounds unchanged. The plateau decline steepens and the nickel mass and
        the velocity rise with :math:`L_{50}`, and a brighter CSM pulse peaks later.
      * **Event class.** ``t_r`` and ``R0`` have one prior for a plain event and another for a CSM event. An event
        without CSM has :math:`L_c=0`.
      * **Fixing a parameter** gives that value to every event, and the trends of the others follow it. ``csm=1`` or
        ``csm=0`` selects the class, which is how the plots below separate the events with CSM from those without.

      The CSM events are part of the Type IIP population and not a separate class, so the rate is that of all Type
      IIP SNe.

      .. rubric:: Calibration

      The data sets used are not equally trustworthy. The Swift and bolometric samples are well-observed events and
      are brighter than the population, so they set shapes and colours, and the ASAS-SN luminosity function sets the
      population's brightness. The priors below are tagged **[LF]** (tuned to the ASAS-SN luminosity function of
      SNe II), **[fit43]** (from fits of the plain model to 43 IIP bolometric light curves), **[csm2]** (from fits to
      SN 2023ixf and SN 2024ggi, :footcite:t:`hiramatsu2023` and :footcite:t:`shrestha2024`), **[lit]** (set from
      the literature) or **[assumed]** (not constrained by the data).

      * **CSM fraction.** ``csm`` is 0.36 [lit], from :footcite:t:`bruch2023` (more than 36%) and
        :footcite:t:`hinds2025` (:math:`36^{+5}_{-7}\%`). ``L_c`` is
        :math:`\log_{10}L_c\sim N(43.00, 0.40)` truncated to :math:`[41.0, 44.8]` [lit], chosen so that the CSM share
        of a magnitude-limited sample approaches that of :footcite:t:`hinds2025`. The luminosity function does not
        constrain it.
      * **Plateau and tail.** ``L50`` is :math:`\log_{10}L_{50}\sim N(41.26, 0.51)` truncated to :math:`[40.8, 43.3]`
        [LF], the faint cutoff being the faintest of the 43 objects. ``t_p``, ``s``, ``a_e``, ``tau_e``, ``t_r`` of
        the plain events and ``M_Ni`` are [fit43].
      * **Photosphere.** ``v_ph`` is [assumed], from :math:`L\propto v^3`. ``T_rec`` is [csm2], with the observed
        temperatures. ``R0`` of the plain events is [assumed], a red supergiant.
      * **CSM pulse.** ``t_c`` and ``tau_d`` are [csm2], ``k`` is [assumed] (SN 2023ixf and SN 2024ggi, with
        :math:`k\ge12`, are the steep tail), and ``R0`` and ``t_r`` of the CSM events are [csm2] and [assumed] (5 to 8
        d for those two objects, the slow tail).
      * **UV blanketing.** ``lam_c``, ``beta_max``, ``T_b`` and ``dT`` were set against the Swift/UVOT light curves
        (see the UV calibration below), on the Swift bands only.

      The model was checked against:

      * the bolometric curves of 43 IIP (plain model: a median 0.02 dex each) and the bolometric curve and
        temperature of SN 2023ixf and SN 2024ggi (0.05 to 0.07 dex in :math:`L`, 0.03 to 0.05 dex in :math:`T`), with
        no UV data used;
      * Swift/UVOT light curves (SOUSA, Brown et al. 2014): at the object level, SN 2012aw, SN 2008in, SN 2024ggi and
        SN 2023ixf (183 detections) scatter 0.20 dex about a +0.02 dex median offset, and at the population level
        (53 SNe II) the median offsets are below 0.2 dex in all three bands from day 0.6 to 15;
      * the ASAS-SN V-band peak luminosity function of SNe II, to 0.06 to 0.10 dex rms over the eight bins from
        :math:`M_V=-15.4` to :math:`-20.0` after fitting one overall normalization. The brightest bin
        (:math:`M_V=-21.2`) is not reproduced and was excluded;
      * a CSM share of a magnitude-limited sample (weights :math:`\propto10^{-0.6M_V}`) of about 60%, against
        :math:`67\pm6\%` from :footcite:t:`hinds2025`.

      The reference fits of the two CSM objects are curve-reproducing sets, not measurements, since the early-excess
      and CSM terms trade off. They are in the model's tests.

      **Limitations**

      * The late-time UV of the population is probably faint, by a factor of 2 to 4 after day 15. Selection effects in
        the Swift sample contribute (the data are 0.3 to 0.65 dex above the model median from day 15 to 110), but
        persistent interaction emission is not modeled.
      * The far UV (below 1900 Å) is uncalibrated, since the blanketing was set on Swift bands only, and there is no
        cut at the Lyman limit.
      * The first day is an extrapolation: no plain IIP has a temperature before day 4 in the calibration set.
      * :math:`t=0` is the explosion, not first light. With CSM, the model can be faint for a short time before an
        observer would record first light.
      * About 17% of the population peaks fainter than the faintest bin of the luminosity function
        (:math:`M_V=-15.4`), and is unconstrained.
      * 37 of the 43 light curves come from a flash-spectroscopy sample, so part of the "plain" early excess is CSM.
      * After about day 80 the temperature of real events can fall below the floor (nebular phase), and the tail
        spectrum is a floor-temperature blackbody. A constant photospheric velocity makes the model cool slightly too
        fast over days 20 to 40 in some objects.
      * There is no host extinction, and the luminosity function was taken at face value.

      .. dropdown:: Parameter priors

         Parameters sampled in :math:`\log_{10}` are written as a normal in the exponent. Luminosities are in erg/s,
         times in days, and the trends are those of the population above.

         .. list-table::
            :header-rows: 1
            :widths: 14 14 42 30

            * - Parameter
              - Symbol
              - Prior
              - Notes
            * - ``csm``
              - :math:`x_\mathrm{csm}`
              - Discrete: 1 with probability 0.36, else 0
              - Whether the event has CSM. A label of the population: the model uses ``L_c``.
            * - ``L50``
              - :math:`L_{50}`
              - Normal(:math:`\log_{10}L_{50}`; mean=41.26, :math:`\sigma`\=0.51), bounds 40.8 to 43.3
              - Plateau luminosity at day 50. [LF]
            * - ``t_p``
              - :math:`t_p`
              - Normal(mean=105 d, :math:`\sigma`\=20 d), bounds 60 to 150 d
              - Plateau duration: the midpoint of the drop. [fit43]
            * - ``s``
              - :math:`s`
              - Normal(mean=0.8 + 0.5(:math:`\log_{10}L_{50}-42`), :math:`\sigma`\=0.5), bounds 0 to 3
              - Plateau decline, in mag per 100 d. [fit43]
            * - ``a_e``
              - :math:`a_e`
              - 4.0 :math:`\times10^{N(0, 0.20)}`, :math:`\log_{10}` bounded to :math:`[-1, 1]`
              - Early excess amplitude, relative to the plateau. [fit43]
            * - ``tau_e``
              - :math:`\tau_e`
              - 10 d :math:`\times10^{N(0, 0.18)}`, :math:`\log_{10}` bounded to :math:`[0.5, 1.5]`
              - Early excess decay time. [fit43]
            * - ``t_r``
              - :math:`t_r`
              - Plain: 1.0 d :math:`\times10^{N(0, 0.35)}`, :math:`\log_{10}` bounded to :math:`[-1, 0.7]`.
                CSM: :math:`\log_{10}(t_r/\mathrm{d})\sim N(\log_{10}1.2, 0.40)`, bounded to :math:`[-0.7, 0.9]`
              - Envelope rise time. [fit43] for plain events, [csm2] and [assumed] for CSM events.
            * - ``M_Ni``
              - :math:`M_\mathrm{Ni}`
              - Normal(:math:`\log_{10}(M_\mathrm{Ni}/M_\odot)`; mean=-1.45 + (:math:`\log_{10}L_{50}-42`),
                :math:`\sigma`\=0.17), bounds -3.3 to -0.5
              - Nickel-56 mass. [fit43]
            * - ``v_ph``
              - :math:`v_\mathrm{ph}`
              - :math:`8\times10^8` cm/s :math:`\times10^{0.33(\log_{10}L_{50}-42)+N(0, 0.08)}`
              - Photospheric expansion velocity. [assumed]
            * - ``R0``
              - :math:`R_0`
              - Plain: :math:`3.5\times10^{13}` cm :math:`\times10^{N(0, 0.15)}`.
                CSM: :math:`1.4\times10^{14}` cm :math:`\times10^{N(0, 0.20)}`, capped at :math:`3\times10^{14}` cm
              - Initial photospheric radius. [assumed] for plain events, [csm2] for CSM events.
            * - ``T_rec``
              - :math:`T_\mathrm{rec}`
              - Normal(mean=6000 K, :math:`\sigma`\=500 K), bounds 4500 to 7000 K
              - Recombination floor temperature. [csm2]
            * - ``L_c``
              - :math:`L_c`
              - Normal(:math:`\log_{10}L_c`; mean=43.00, :math:`\sigma`\=0.40), bounds 41.0 to 44.8
              - CSM pulse peak luminosity (0 for a plain event). [lit]
            * - ``t_c``
              - :math:`t_c`
              - Normal(:math:`\log_{10}(t_c/\mathrm{d})`; mean=:math:`\log_{10}3.5+0.5(\log_{10}L_c-43.15)`,
                :math:`\sigma`\=0.15), bounds 0 to 1
              - CSM pulse peak time. [csm2]
            * - ``k``
              - :math:`k`
              - 4.0 :math:`\times10^{N(0, 0.30)}`, :math:`\log_{10}` bounded to :math:`[0.18, 1.18]`
              - CSM pulse rise index. [assumed]
            * - ``tau_d``
              - :math:`\tau_d`
              - 1.5 d :math:`\times10^{N(0, 0.30)}`, :math:`\log_{10}` bounded to :math:`[-0.5, 1.0]`
              - CSM pulse decay time. [csm2]
            * - ``w``
              - :math:`w`
              - Fixed (5 d)
              - Width of the plateau drop.
            * - ``T0``
              - :math:`T_0`
              - Fixed (400 d)
              - Gamma-ray trapping timescale.
            * - ``f_c``
              - :math:`f_c`
              - Fixed (1.25)
              - Colour to effective temperature ratio.
            * - ``n``
              - :math:`n`
              - Fixed (6)
              - Smoothness of the transition to the temperature floor.
            * - ``lam_c``
              - :math:`\lambda_c`
              - Fixed (3975 Å)
              - Blanketing cutoff wavelength.
            * - ``beta_max``
              - :math:`\beta_\mathrm{max}`
              - Fixed (3.6)
              - Full-strength suppression index.
            * - ``T_b``
              - :math:`T_b`
              - Fixed (8000 K)
              - Temperature at which the blanketing is half on.
            * - ``dT``
              - :math:`\Delta T`
              - Fixed (660 K)
              - Width of the blanketing turn-on.

      .. rubric:: Simulated Light Curves

      The plot below draws 1000 random parameter realizations from the priors above for each of the two classes,
      the events without CSM (``csm=0``) and with it (``csm=1``), and shows the median, 68% and 95% envelopes of
      their bolometric light curves and photospheric temperatures. The data are the bolometric light curves and
      temperatures of the literature Type IIP SNe :footcite:p:`dallora2014,faran2018`, including the 35 "comparison"
      SNe II of :footcite:t:`jacobsongalan2024`, in blue, and SN 2023ixf :footcite:p:`hsu2025` and SN 2024ggi
      :footcite:p:`chen2024`, the two events with CSM in the archive, in red. The luminosity scale was set against the
      ASAS-SN luminosity function (see the calibration above), not fit to these curves, which are brighter than the
      population. The events that are not SN 2023ixf and SN 2024ggi are not known to be free of CSM, and
      :footcite:t:`bruch2023` find that a large fraction of the sample has early interaction signatures.

      .. plot::
         :include-source: false

         import numpy as np
         import matplotlib.pyplot as plt
         from astropy import units as u

         from uvex_transients.models.supernovae import TypeIIPSED
         from uvex_transients.utils.lightcurve_archive import LightcurveArchive

         n_samples = 1000
         colors = {"plain": "#2166ac", "csm": "#d6604d"}
         t = np.geomspace(0.3, 200, 300) * u.day
         t_days = t.to_value(u.day)

         sims = {}
         for key, csm, seed in [("plain", 0.0, 1), ("csm", 1.0, 2)]:
             params = TypeIIPSED(csm=csm).sample_parameters(size=n_samples, rng=np.random.default_rng(seed))
             params_grid = {name: value[:, None] for name, value in params.items()}
             sims[key] = {
                 "L": TypeIIPSED.eval_bolometric(t, **params_grid).to_value(u.erg / u.s),
                 "T": TypeIIPSED.temperature(t, **params_grid).to_value(u.K),
             }

         fig, (ax_L, ax_T) = plt.subplots(2, 1, figsize=(6.4, 7.6), sharex=True)
         for key, label in [("plain", "No CSM"), ("csm", "CSM")]:
             for ax, quantity in [(ax_L, "L"), (ax_T, "T")]:
                 lo95, lo68, med, hi68, hi95 = np.percentile(sims[key][quantity], [2.5, 16, 50, 84, 97.5], axis=0)
                 ax.fill_between(t_days, lo95, hi95, color=colors[key], alpha=0.15, lw=0)
                 ax.fill_between(t_days, lo68, hi68, color=colors[key], alpha=0.35, lw=0)
                 ax.plot(
                     t_days, med, color=colors[key], lw=2,
                     label=f"{label} model: median, 68%, 95% (n={n_samples})" if quantity == "L" else None,
                 )

         archive = LightcurveArchive()
         observed = {
             "plain": ("supernovae/IIP", "o", 14, "Literature Type IIP"),
             "csm": ("supernovae/II", "s", 22, "SN 2023ixf, SN 2024ggi (CSM)"),
         }
         for key, (group, marker, size, label) in observed.items():
             first = True
             for event in archive.events(group):
                 for ax, field in [(ax_L, "L_bol"), (ax_T, "T_phot")]:
                     try:
                         table = archive.table(group, event, field)
                     except Exception:
                         continue
                     good = np.isfinite(table[field].value) & (table[field].value > 0)
                     ax.scatter(
                         table["time"].to_value(u.day)[good], table[field].value[good],
                         marker=marker, s=size, color=colors[key], edgecolor="white", linewidth=0.3, zorder=5,
                         label=label if (first and field == "L_bol") else None,
                     )
                 first = False

         ax_L.set_xscale("log")
         ax_L.set_yscale("log")
         ax_L.set_ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
         ax_L.set_title("Type IIP SNe with and without CSM: simulated light curves")
         ax_L.legend(loc="lower left", fontsize=8, frameon=False)
         ax_L.set_ylim(1e40, 3e44)
         ax_L.set_xlim(0.3, 200)
         ax_T.set_yscale("log")
         ax_T.set_xlabel("Time since explosion [days]")
         ax_T.set_ylabel("Photospheric temperature [K]")
         ax_T.set_ylim(3e3, 8e4)

         fig.tight_layout()

      The events with CSM are shown on their own below, 300 draws with ``csm=1``, against SN 2023ixf
      :footcite:p:`hsu2025` and SN 2024ggi :footcite:p:`chen2024`. The CSM pulse dominates the first days, and after
      it the events follow the plateau and tail of the others.

      .. plot::
         :include-source: false

         import numpy as np
         import matplotlib.pyplot as plt
         from astropy import units as u

         from uvex_transients.models.supernovae import TypeIIPSED
         from uvex_transients.utils.lightcurve_archive import LightcurveArchive

         rng = np.random.default_rng(20260912)
         n_samples = 300

         params = TypeIIPSED(csm=1.0).sample_parameters(size=n_samples, rng=rng)
         params_grid = {name: value[:, None] for name, value in params.items()}

         t = np.geomspace(0.3, 200, 400) * u.day
         L_bol = TypeIIPSED.eval_bolometric(t, **params_grid)
         T = TypeIIPSED.temperature(t, **params_grid)

         fig, (ax_L, ax_T) = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True)
         for row in range(n_samples):
             ax_L.plot(t.to_value(u.day), L_bol[row].to_value(u.erg / u.s), color="C0", lw=0.4, alpha=0.2)
             ax_T.plot(t.to_value(u.day), T[row].to_value(u.K), color="C3", lw=0.4, alpha=0.2)

         archive = LightcurveArchive()
         for suffix, label, marker, color in [
             ("2023ixf_hsu2025", "SN 2023ixf", "o", "k"),
             ("2024ggi_chen2024", "SN 2024ggi", "s", "firebrick"),
         ]:
             lbol_obs = archive.table("supernovae/II", suffix, "L_bol")
             Tphot_obs = archive.table("supernovae/II", suffix, "T_phot")
             ax_L.scatter(
                 lbol_obs["time"].to_value(u.day), lbol_obs["L_bol"].to_value(u.erg / u.s),
                 marker=marker, s=28, color=color, edgecolor="white", linewidth=0.5, zorder=5, label=label,
             )
             ax_T.scatter(
                 Tphot_obs["time"].to_value(u.day), Tphot_obs["T_phot"].to_value(u.K),
                 marker=marker, s=28, color=color, edgecolor="white", linewidth=0.5, zorder=5, label=label,
             )

         ax_L.set_xscale("log")
         ax_L.set_yscale("log")
         ax_L.set_ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
         ax_L.set_title(f"Type IIP SNe with CSM: simulated light curves (n={n_samples})")
         ax_L.legend(loc="lower left", fontsize=8, frameon=False)
         ax_L.set_ylim([1e40, 3e44])
         ax_L.set_xlim(0.3, 200)
         ax_T.set_yscale("log")
         ax_T.set_xlabel("Time since explosion [days]")
         ax_T.set_ylabel("Photospheric temperature [K]")
         ax_T.set_ylim([3e3, 6e4])

         fig.tight_layout()

      .. rubric:: UV Calibration

      The plots below overlay the Swift/UVOT UVW1 (2600 Å) and UVW2 (1928 Å) light curves of
      Type IIP supernovae from the Swift Optical/Ultraviolet Supernova Archive (SOUSA) on 100 random draws from the priors above,
      integrated through the corresponding UVOT transmission curve from the SVO Filter Profile Service. Both are shown as
      :math:`\nu L_\nu` in the observed band with no K-correction. Archive supernovae are drawn at random, and the
      legend gives how many are shown. Only those with an estimated explosion date (black) are timed from explosion;
      the rest (grey) are timed from discovery, so they sit to the left of where the model places the same phase, by
      an amount that differs from event to event.

      11 Type IIP supernovae have at least four UVW1 detections, and 9 have at least four UVM2 or UVW2 detections.
      The blanketing parameters ``lam_c``, ``beta_max``, ``T_b`` and ``dT`` were set against these light curves, by
      comparing the model's peak :math:`\nu L_\nu`, the UVW1 minus UVW2 colour and the late-time level with the data.
      These are ordinary Type IIP, so the comparison is with the model's events without CSM (``csm=0``). At the
      object level the model scatters 0.20 dex about the data, and at the population level the median offsets are
      below 0.2 dex in all three bands from day 0.6 to 15. The Swift supernovae are well observed and so brighter
      than the population, so the model sits below their medians. The late-time UV is probably faint in the model
      (see the limitations above), and the UVOT bands do not reach the UVEX FUV band, so the UV there is an
      extrapolation.

      .. plot::
         :include-source: false

         import matplotlib.pyplot as plt

         from uvex_transients.models.supernovae import TypeIIPSED
         from uvex_transients.utils.plotting import plot_swift_uv_calibration

         plot_swift_uv_calibration(TypeIIPSED, "IIP", band="UVW1", fixed={"csm": 0.0})
         plt.show()

      .. plot::
         :include-source: false

         import matplotlib.pyplot as plt

         from uvex_transients.models.supernovae import TypeIIPSED
         from uvex_transients.utils.plotting import plot_swift_uv_calibration

         plot_swift_uv_calibration(TypeIIPSED, "IIP", band="UVW2", fixed={"csm": 0.0})
         plt.show()

      The events with CSM (``csm=1``) are compared with the two Swift supernovae of that kind (SN 2023ixf and SN
      2024ggi, timed from explosion), below. The UV was not fit. The Swift light curves of SN 2023ixf begin at 24 d,
      so it constrains only the late UV.

      .. plot::
         :include-source: false

         import matplotlib.pyplot as plt

         from uvex_transients.models.supernovae import TypeIIPSED
         from uvex_transients.utils.plotting import plot_swift_uv_calibration

         plot_swift_uv_calibration(TypeIIPSED, "IIP+Excess", band="UVW1", fixed={"csm": 1.0})
         plt.show()

      .. plot::
         :include-source: false

         import matplotlib.pyplot as plt

         from uvex_transients.models.supernovae import TypeIIPSED
         from uvex_transients.utils.plotting import plot_swift_uv_calibration

         plot_swift_uv_calibration(TypeIIPSED, "IIP+Excess", band="UVW2", fixed={"csm": 1.0})
         plt.show()

      .. rubric:: Luminosity Function

      The distribution of peak absolute magnitude in the Johnson V band, from 10000 draws of the SED
      priors above. Each draw's rest-frame band light curve is measured at its brightest point. The curve
      is normalized per event, so it integrates to 1 over :math:`M_V`: multiply it by the rate to get a
      volumetric luminosity function, in events per comoving volume, time and magnitude. The shaded
      bands are the 68% (Clopper-Pearson) intervals of each bin, and bins with no events are not drawn.
      :math:`M_V` is the AB magnitude at 10 pc of a spectrum flat in :math:`L_\nu` that has the same
      Johnson V band luminosity, a fixed conversion of the luminosity rather than the photon-weighted
      magnitude of the SED itself.

      The curve is of the whole population, with and without CSM, and the points are the luminosity
      function of SNe II of ASAS-SN, which the ``L50`` prior was set against (so this is a check of that choice, not
      an independent test). Its rate has a normalization that is not ours, so
      only its shape is compared: the points are scaled to unit area over the magnitudes they cover and then by the
      fraction of the model's events in that range. Where the shapes agree, the points lie on the model's curve.
      The ASAS-SN class includes all hydrogen-rich types, not only the plateau events, and its points are the bright
      side of a magnitude-limited sample, so the faint end of the model is not tested.

      .. plot::
         :include-source: false

         import matplotlib.pyplot as plt

         from uvex_transients.transients.supernovae import TypeIIPSNe
         from uvex_transients.utils.filter_utils import get_bandpass
         from uvex_transients.utils.plotting import plot_archived_luminosity_function, plot_luminosity_function

         v_band = get_bandpass("Generic/Johnson.V")
         lf = TypeIIPSNe().luminosity_function(v_band, 10000, quantity="magnitude", bins=30, rng=0)
         fig, ax = plot_luminosity_function(lf, band="V", label="Model")
         plot_archived_luminosity_function(lf, "supernovae/II", "ASAS-SN", ax=ax)
         plt.show()

      .. rubric:: Observability Summary

      The plots below show how the redshift limit follows from the SED. We draw 1000 sets of
      parameters from the priors above and evaluate each of them across a grid of redshifts, in the
      UVEX FUV and NUV bands, over the whole rest-frame light curve. The left panel shows each
      draw's peak apparent AB magnitude, in whichever band is brighter, against redshift: a draw is
      detectable out to the redshift where its line crosses the UVEX 1 Dwell limit of
      :math:`m<24.5`. The right panel shows the redshift beyond which, with 95% confidence, at most
      1% of the population is detectable, as a function of the magnitude limit. The redshift limit
      adopted above, :math:`z=1.07`, is its value at :math:`m=24.5`, rounded up and derived on a
      finer grid than this plot's.

      The limit bounds the fraction of the *population* that is detectable beyond it, not the
      fraction of detected events, and it does not include Milky Way extinction or sky position.

      This limit is an extrapolation. It comes from the hot early photosphere, whose UV the model does not blanket
      while it is above :math:`\sim10^4` K, and from the CSM events, whose pulse is bright and hot. At
      :math:`z\approx1.07` the UVEX FUV band (1333 to 1817 Å) samples rest-frame wavelengths of about 640 to 880 Å,
      shortward of the hydrogen Lyman limit, and the NUV band (2050 to 2735 Å) about 990 to 1320 Å. The model has no
      cutoff at the Lyman limit, and no data constrain it below the 1900 Å rest-frame edge of the Swift/UVOT bands,
      so where a real spectrum is cut off would lower the limit. The model has no host dust. The previous limit of
      :math:`z=0.51` (0.53 for the separate excess population this model replaces) came from a model whose UV was
      suppressed at all epochs, so the increase is a property of the new SED, not a measurement.

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

         The shock cooling priors are those of :class:`~uvex_transients.models.shock_cooling.MoragShockCoolingSED`,
         except that ``radius`` and ``envelope_mass`` are wider (see the UV calibration below), and the nickel priors are those of the Type Ib SED
         (:class:`~uvex_transients.models.supernovae.Ibc.TypeIbSED`, without its host dust). Neither is a fit to Type IIb events, except that the
         nickel mass was set against the ASAS-SN luminosity function of SNe IIb (see the luminosity function below), and
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
           - :math:`z = 1.25`
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
         line-blanketing parameters those of :class:`~uvex_transients.models.supernovae.Ibc.TypeIbSED` (which
         has host dust, that this model does not), except ``M_Ni``. The line blanketing is that of Type Ib, set against
         Swift/UVOT (see the UV calibration below), and applies to the nickel component only: the shock cooling
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
              - LogNormal(:math:`\sigma`\=1.1) around :math:`6\times10^{12}` cm
              - Progenitor stellar radius. Wider than the Morag value (0.85): see the UV calibration below.
            * - ``opacity``
              - :math:`\kappa`
              - Fixed (0.34 cm\ :sup:`2` g\ :sup:`-1`)
              - Electron-scattering opacity of the shock cooling.
            * - ``envelope_mass``
              - :math:`M_E`
              - LogNormal(:math:`\sigma`\=1.3) around :math:`0.05\ M_\odot`
              - Hydrogen-rich envelope mass. Wider than the Morag value (1.0): see the UV calibration below.
            * - ``core_mass``
              - :math:`M_C`
              - LogNormal(:math:`\sigma`\=0.35) around :math:`2.8\ M_\odot`
              - Core mass.
            * - ``M_Ni``
              - :math:`M_\mathrm{Ni}`
              - LogNormal(median 0.02 :math:`M_\odot`, :math:`\sigma_{\ln}`\=1.1), bounds 0.005 to 1.5 :math:`M_\odot`
              - Nickel-56 mass, set against the ASAS-SN luminosity function of SNe IIb. It is below the Type Ib median
                (0.06 :math:`M_\odot`), and was not checked against measured Type IIb nickel masses: see the
                luminosity function below.
            * - ``M_ej``
              - :math:`M_\mathrm{ej}`
              - TruncatedNormal(mean=2.3, :math:`\sigma`\=2.2, bounds 0.1 to 20 :math:`M_\odot`)
              - Ejecta mass.
            * - ``v_ej``
              - :math:`v_\mathrm{ej}`
              - TruncatedNormal(mean=9.6, :math:`\sigma`\=3.5, bounds 4 to 25; units of :math:`10^3` km/s)
              - Constant ejecta velocity.
            * - ``kappa``
              - :math:`\kappa`
              - Fixed (0.07 cm\ :sup:`2` g\ :sup:`-1`)
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
              - Fixed (:math:`1.4\times10^{15}` Hz)
              - Frequency blueward of which line blanketing suppresses the nickel component. The Type Ib value.
            * - ``beta``
              - :math:`\beta`
              - Fixed (3)
              - Power-law index of the line-blanketing suppression. The Type Ib value.

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

      Only the nickel component of this SED is line-blanketed, with the values set for Type Ib (not refit to Type IIb), so the UV in the first days is
      set by the unblanketed shock cooling component. 7 Type IIb supernovae have at least four UVW1 detections, and
      6 have at least four UVW2 detections.

      Nine Type IIb supernovae have Swift detections, all timed from discovery. The peak :math:`\nu L_\nu` of each, in
      the model's distribution of peaks at :math:`z=0.01` (taken over the epochs the supernova was observed), has a
      median percentile of about 40% in UVW1, so the model is not low overall. The events span 3 dex in peak
      luminosity. The ``radius`` and ``envelope_mass`` priors of the shock cooling were widened (from 0.85 and 1.0 to
      1.1 and 1.3 in :math:`\sigma_{\ln}`) to widen the model's 95% range of UV peaks from 2.5 dex to 2.9 dex. SN 2013cu,
      which has the brightest peak (:math:`\log\nu L_\nu=43.2` in UVW1), remains at about the 99th percentile of the model,
      which one event in nine is unlikely to do by chance, so the model's bright tail is probably still too short. The
      Type Ib values of the line blanketing were not refit to these events, and
      the data are too few to constrain it.

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

      .. rubric:: Luminosity Function

      The distribution of peak absolute magnitude in the Johnson V band, from 10000 draws of the SED
      priors above. Each draw's rest-frame band light curve is measured at its brightest point. The curve
      is normalized per event, so it integrates to 1 over :math:`M_V`: multiply it by the rate to get a
      volumetric luminosity function, in events per comoving volume, time and magnitude. The shaded
      bands are the 68% (Clopper-Pearson) intervals of each bin, and bins with no events are not drawn.
      :math:`M_V` is the AB magnitude at 10 pc of a spectrum flat in :math:`L_\nu` that has the same
      Johnson V band luminosity, a fixed conversion of the luminosity rather than the photon-weighted
      magnitude of the SED itself.

      The points are the luminosity function of SNe IIb of ASAS-SN, which the ``M_Ni`` prior was set against (so this
      is a check of that choice, not an independent test). They are absolute rates, and the integrated rate over them,
      :math:`9.5\times10^{-6}` SNe/Mpc\ :sup:`3`/yr, is within 12% of the rate adopted above, so the plot shows them
      scaled by shape, as for Type IIP, but no normalization was fit. The nickel mass is the only parameter
      that matters here: it has a correlation of 0.8 with the peak luminosity, and the shock cooling and ejecta
      mass parameters are weak knobs. There are seven points, with fractional errors that are not given, and they are
      not smooth (the :math:`M_V=-17.4` point lies below both neighbours), so the fit is limited to about 0.35 dex rms
      by the data. It prefers median nickel masses of 0.01 to 0.02 :math:`M_\odot` with a wide scatter, and the adopted
      0.02 :math:`M_\odot` is the upper end of what the data allow. That is below the Type Ib median (0.06
      :math:`M_\odot`) and the 0.1 :math:`M_\odot` that this model used before, and it was not checked against
      nickel masses measured for Type IIb events. The model has too many events at :math:`M_V=-16.1` and too few
      at :math:`M_V=-15.4`, which has most of the observed rate.

      .. plot::
         :include-source: false

         import matplotlib.pyplot as plt

         from uvex_transients.transients.supernovae import TypeIIbSNe
         from uvex_transients.utils.filter_utils import get_bandpass
         from uvex_transients.utils.plotting import plot_archived_luminosity_function, plot_luminosity_function

         v_band = get_bandpass("Generic/Johnson.V")
         lf = TypeIIbSNe().luminosity_function(v_band, 10000, quantity="magnitude", bins=30, rng=0)
         fig, ax = plot_luminosity_function(lf, band="V", label="Model")
         plot_archived_luminosity_function(lf, "supernovae/IIb", "ASAS-SN", ax=ax)
         plt.show()

      .. rubric:: Observability Summary

      The plots below show how the redshift limit follows from the SED. We draw 1000 sets of
      parameters from the priors above and evaluate each of them across a grid of redshifts, in the
      UVEX FUV and NUV bands, over the whole rest-frame light curve. The left panel shows each
      draw's peak apparent AB magnitude, in whichever band is brighter, against redshift: a draw is
      detectable out to the redshift where its line crosses the UVEX 1 Dwell limit of
      :math:`m<24.5`. The right panel shows the redshift beyond which, with 95% confidence, at most
      1% of the population is detectable, as a function of the magnitude limit. The redshift limit
      adopted above, :math:`z=1.25`, is its value at :math:`m=24.5`, rounded up and derived on a
      finer grid than this plot's. Epochs earlier than 0.1 d are excluded from the light curve, because the
      shock cooling luminosity of this SED grows without bound toward :math:`t\to0`.

      The limit is set by the brightest early shock cooling peaks, which the Swift data constrain only weakly (see the UV
      calibration above), and it rose from 0.79 to 1.25 when the ``radius`` and ``envelope_mass`` priors were widened.
      It is an extrapolation.

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
