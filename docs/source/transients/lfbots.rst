.. _transients_lfbots:

Luminous Fast Blue Optical Transients
========================================

Luminous fast blue optical transients (LFBOTs), typified by the prototype AT2018cow, are modeled
here as a rapidly-evolving, cooling blackbody photosphere -- a phenomenological description of
their fast rise/decline (:math:`\lesssim10` day timescales), persistently blue colors, and, in
several cases, non-thermal emission pointing to a central engine rather than radioactive decay
:footcite:p:`holu2026`.

This population is implemented by
:class:`~uvex_transients.transients.LFBOTs.LuminousFastBlueOpticalTransient`, pairing
:class:`~uvex_transients.models.lfbots.lfbots.LFBOTCoolingBlackbodySED` with the rate/duration
metadata described below.

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
     - :math:`10\ \mathrm{Gpc^{-3}\,yr^{-1}}` (constant in :math:`z`)
     - :footcite:t:`perley2026,holu2026`
     - Reported LFBOT rates span more than two orders of magnitude across differing selection
       criteria: :footcite:t:`coppejans2020` find :math:`<300\ \mathrm{Gpc^{-3}\,yr^{-1}}` (PTF) and
       :math:`700`-:math:`1400\ \mathrm{Gpc^{-3}\,yr^{-1}}` (PS1-MDS); the delayed-dynamical-instability
       model of :footcite:t:`klencki2025` predicts :math:`15`-:math:`300\ \mathrm{Gpc^{-3}\,yr^{-1}}`;
       :footcite:t:`perley2026` and :footcite:t:`holu2026` report the lowest rates,
       :math:`0.9`-:math:`12.5\ \mathrm{Gpc^{-3}\,yr^{-1}}`, which is adopted here as
       :attr:`~uvex_transients.transients.LFBOTs.LuminousFastBlueOpticalTransient.RATE_CI` (see the
       dropdown below for why, and :ref:`user_guide_transients_rate_uncertainty` for the general
       mechanism). Taken as constant in :math:`z`, since LFBOTs are too rare for their redshift
       evolution to yet be meaningfully constrained.
   * - Redshift limit
     - :math:`z = 3`
     - --
     - Chosen to safely bound the population relative to the SED's own (much shorter) intrinsic
       rise/decline timescales.
   * - Duration
     - 100 days
     - --
     - Generous relative to the SED's own rise/decline timescales, to safely bound the slowly
       fading power-law tail.

.. dropdown:: Rate uncertainty: CLU vs. BTS

   Volumetric LFBOT rates in the literature come from two different ZTF-era surveys, with
   different selection functions, that nonetheless both count AT2018cow itself among their
   events:

   - The **Census of the Local Universe (CLU)** experiment is *volume-limited*: it targets
     galaxies out to a fixed distance regardless of apparent brightness, so it is more complete
     for intrinsically fainter events but probes a smaller total volume.
   - The **Bright Transient Survey (BTS)** is *magnitude-limited*: it is complete down to a fixed
     apparent magnitude regardless of distance, so it reaches a much larger volume but is biased
     against faint/fast-fading events -- exactly the population LFBOTs belong to.

   :footcite:t:`ho2023` derive rates from both: :math:`2.7`-:math:`546\ \mathrm{Gpc^{-3}\,yr^{-1}}`
   from CLU, and :math:`0.31`-:math:`85.3\ \mathrm{Gpc^{-3}\,yr^{-1}}` from BTS. Both ranges are
   wide because each survey's LFBOT sample is small (a handful of events), so the rate is
   dominated by Poisson/small-number uncertainty rather than by the selection function itself.
   :footcite:t:`perley2026` later revise the BTS estimate down to a narrower
   :math:`0.9`-:math:`12.5\ \mathrm{Gpc^{-3}\,yr^{-1}}`, using a substantially larger BTS-selected
   sample than :footcite:t:`ho2023`'s original estimate.

   This population adopts :footcite:t:`perley2026`'s revised BTS range, rather than
   :footcite:t:`ho2023`'s original BTS range or either paper's CLU range, purely because it is
   built from more events and is therefore the most statistically robust of the three -- not
   because BTS's magnitude-limited selection is judged astrophysically preferable to CLU's
   volume-limited one.

SED Model
----------

:class:`~uvex_transients.models.lfbots.lfbots.LFBOTCoolingBlackbodySED` pairs a Gaussian-rise/power-law
decline bolometric light curve
(:class:`~uvex_transients.models.lightcurves.generic.GaussianRisePowerLawLightcurve`) with a
cooling blackbody photosphere,

.. math::

    L_\mathrm{bol}(t) = A \times
    \begin{cases}
        \exp\left[-\dfrac{(t-t_\mathrm{peak})^2}{2(t_\mathrm{peak}/5)^2}\right], & t \le t_\mathrm{peak} \\[6pt]
        (t/t_\mathrm{peak})^{-\alpha_\mathrm{decline}}, & t > t_\mathrm{peak}
    \end{cases}, \qquad
    T(t) = T_\mathrm{floor} + (T_0 - T_\mathrm{floor})\left(1 + \frac{t}{t_\mathrm{peak}}\right)^{-\alpha_T}.

The cooling law has the same functional form used for the other cooling-blackbody SEDs in this
codebase (see :class:`~uvex_transients.models.supernovae.villar.VillarCoolingBlackbodySED`), but with the
cooling timescale fixed to the light curve's own peak time :math:`t_\mathrm{peak}` rather than left
as a separate free parameter, since LFBOT light curves are too fast and too sparsely sampled to
usefully distinguish the two. This choice, together with the default priors below, follows the
bolometric light curves :footcite:t:`holu2026` compile for six known LFBOTs: a late-time
:math:`L_\mathrm{bol}\propto t^{-3}` decline with peaks shorter than about 10 days (CSS161010 and
"puz"), a peak bolometric luminosity ranging from :math:`10^{44}\ \mathrm{erg\,s^{-1}}`
(CSS161010) to around :math:`2\times10^{45}\ \mathrm{erg\,s^{-1}}` ("wpp"), and a blackbody
temperature evolving post-peak roughly as :math:`T\sim t^{-1/3}` (CSS161010 was closer to constant
temperature; AT2018cow and "qfm" were both :math:`\sim t^{-1/3}`) :footcite:p:`holu2026`.

.. list-table::
   :header-rows: 1
   :widths: 16 12 40 32

   * - Parameter
     - Symbol
     - Prior
     - Notes / Source
   * - ``amplitude``
     - :math:`A`
     - Normal(:math:`\log_{10}(L_0/\mathrm{erg\,s^{-1}})`; mean=44.5, :math:`\sigma`\=0.5)
     - Peak bolometric luminosity, spanning the :math:`\sim10^{44}`-few :math:`\times10^{45}`
       erg/s range :footcite:p:`holu2026`.
   * - ``t_peak``
     - :math:`t_\mathrm{peak}`
     - LogNormal(:math:`\log_{10}(t_\mathrm{peak}/\mathrm{d})`; mean=:math:`\log_{10}3`, :math:`\sigma`\=0.1)
     - Time of peak bolometric luminosity, log-normal around a few days with small scatter.
   * - ``decline_index``
     - :math:`\alpha_\mathrm{decline}`
     - Uniform(2, 4)
     - Post-peak decline index; late-time light curves broadly consistent with :math:`t^{-3}`
       :footcite:p:`holu2026`.
   * - ``T0``
     - :math:`T_0`
     - LogNormal(:math:`T_0/2\times10^4\,\mathrm{K}`; mean=0, :math:`\sigma`\=0.15)
     - Photospheric temperature at :math:`t=0` (the :math:`T(t)\to T_0` limit).
   * - ``T_floor``
     - :math:`T_\mathrm{floor}`
     - LogNormal(:math:`T_\mathrm{floor}/10^4\,\mathrm{K}`; mean=0, :math:`\sigma`\=0.01)
     - Asymptotic late-time temperature, tightly centered on :math:`10^4` K.
   * - ``alpha_T``
     - :math:`\alpha_T`
     - Uniform(0, 1/3)
     - Late-time cooling index; :math:`\sim1/3` is a reasonable fit for most events (e.g.
       AT2018cow) :footcite:p:`holu2026`.

Simulated Light Curves
~~~~~~~~~~~~~~~~~~~~~~~

The plot below draws 1000 random parameter realizations from the priors above and shows the
resulting bolometric light curves and photospheric temperatures, against the observed
bolometric light curves and photospheric temperatures of three known LFBOTs -- AT2018cow,
AT2024wpp, and AT2024puz -- compiled by :footcite:t:`holu2026`.

.. plot::
   :include-source: false

   import numpy as np
   import matplotlib.pyplot as plt
   from astropy import units as u

   from uvex_transients.models.lfbots import LFBOTCoolingBlackbodySED as SEDClass
   from uvex_transients.utils.lightcurve_archive import LightcurveArchive

   rng = np.random.default_rng(20260910)
   n_samples = 1000

   params = SEDClass().sample_parameters(size=n_samples, rng=rng)
   params_grid = {name: value[:, None] for name, value in params.items()}

   t = np.geomspace(0.05, 100, 200) * u.day
   L_bol = SEDClass.eval_bolometric(t, **params_grid)
   T = SEDClass.temperature(t, **params_grid)

   fig, (ax_L, ax_T) = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True)

   for row in range(n_samples):
       ax_L.plot(t.to_value(u.day), L_bol[row].to_value(u.erg / u.s), color="C0", lw=0.4, alpha=0.06)
       ax_T.plot(t.to_value(u.day), T[row].to_value(u.K), color="C3", lw=0.4, alpha=0.06)

   archive = LightcurveArchive()
   observed_lfbots = [
       ("2018cow_holu2026", "AT2018cow (Ho & Lu+2026)", "o", "k"),
       ("2024wpp_holu2026", "AT2024wpp (Ho & Lu+2026)", "^", "darkorange"),
       ("2024puz_holu2026", "AT2024puz (Ho & Lu+2026)", "D", "seagreen"),
   ]

   for suffix, label, marker, color in observed_lfbots:
       lbol_obs = archive.table("lfbots", suffix, "L_bol")
       Tphot_obs = archive.table("lfbots", suffix, "T_phot")
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
   ax_L.set_title("LFBOTs: simulated bolometric light curves (n=1000)")
   ax_L.legend(loc="upper right", fontsize=8, frameon=False)

   ax_T.set_yscale("log")
   ax_T.set_xlabel("Time since explosion [days]")
   ax_T.set_ylabel("Photospheric temperature [K]")

   fig.tight_layout()


----

Observability Summary
----------------------

Below are the redshifts :math:`z` and corresponding bandpass calculated peak apparent AB magnitudes
:math:`m_\mathrm{AB}` of 3000 simulated LFBOTs drawn from the priors above, with
the UVEX 1 Dwell limit of :math:`m<24.5` overplotted. Findings here justify our confidence in a
:math:`z=3` redshift limit for this population.

.. plot::
   :include-source: false

    import numpy as np
    import matplotlib.pyplot as plt
    from astropy import units as u
    from scipy.stats import gaussian_kde

    from m4opt.missions import uvex
    from uvex_transients.transients.LFBOTs import LuminousFastBlueOpticalTransient
    from uvex_transients.models.lfbots import LFBOTCoolingBlackbodySED as SEDClass

    rng = np.random.default_rng(20260911)
    n_samples = 3000

    lfbot = LuminousFastBlueOpticalTransient()
    z = lfbot.sample_event_redshift(n_samples, rng=rng)
    params = SEDClass().sample_parameters(size=n_samples, rng=rng)

    # Observed-frame time of rest-frame peak, i.e. where each event is brightest as seen by UVEX.
    t_obs_peak = params["t_peak"] * (1.0 + z)

    bandpasses = uvex.detector.bandpasses
    band_names = list(bandpasses)

    fig, axes = plt.subplots(1, len(band_names), figsize=(10.5, 4.8), sharey=True)

    for ax, band_name in zip(axes, band_names):
        mag = SEDClass.mag_bandpass(bandpasses[band_name], t_obs_peak, redshift=z, **params).to_value(u.ABmag)
        finite = np.isfinite(mag)
        z_finite, mag_finite = z[finite], mag[finite]

        ax.scatter(z_finite, mag_finite, s=5, ec='k',fc='k',alpha=0.5, label="Simulated events")

        kde = gaussian_kde(np.vstack([z_finite, mag_finite]))
        z_grid = np.linspace(z_finite.min(), z_finite.max(), 150)
        mag_grid = np.linspace(mag_finite.min(), mag_finite.max(), 150)
        Z_grid, Mag_grid = np.meshgrid(z_grid, mag_grid)
        density = kde(np.vstack([Z_grid.ravel(), Mag_grid.ravel()])).reshape(Z_grid.shape)
        ax.contour(Z_grid, Mag_grid, density, levels=6, colors="k", linewidths=0.7)

        ax.axhline(24.5, color="firebrick", ls="--", lw=1.2, label="UVEX limit (1 Dwell)")

        ax.invert_yaxis()
        ax.set_xlabel("Redshift")
        ax.set_title(f"UVEX {band_name}")
        ax.legend(loc="upper right", fontsize=8, frameon=False)

        ax.invert_yaxis()
        ax.set_ylim([50, 15])

    axes[0].set_ylabel("Peak apparent AB magnitude")
    fig.suptitle(f"LFBOTs: peak apparent magnitude vs. redshift (n={n_samples})")
    fig.tight_layout()
    plt.show()

The anticipated rate of LFBOTs detectable by UVEX at these limits is as follows assuming that
any event above the :math:`m<24.5` limit is detectable, and that the population is isotropic and homogeneous
in comoving volume out to :math:`z=3`:

.. plot::
   :include-source: false

    import numpy as np
    import matplotlib.pyplot as plt
    from astropy import units as u

    from m4opt.missions import uvex
    from uvex_transients.transients.LFBOTs import LuminousFastBlueOpticalTransient


    rng = np.random.default_rng(20260911)
    n_samples = 3000

    # Sample the LFBOT population.
    lfbot = LuminousFastBlueOpticalTransient()
    redshift = lfbot.sample_event_redshift(n_samples, rng=rng)
    params = lfbot.sed.sample_parameters(size=n_samples, rng=rng)

    # The all-sky rate, with no survey footprint applied.
    all_sky_rate = lfbot.all_sky_rate

    # Observed-frame time corresponding to the rest-frame peak.
    t_peak_obs = params["t_peak"] * (1 + redshift)

    detection_limits = {
        "FUV": 24.5 * u.ABmag,
        "NUV": 24.5 * u.ABmag,
    }

    # Compute peak-visible rates.
    visible_rates = {}

    for band_name, bandpass in uvex.detector.bandpasses.items():
        magnitudes = lfbot.sed.mag_bandpass(
            bandpass,
            t_peak_obs,
            redshift=redshift,
            **params,
        ).to_value(u.ABmag)

        visible = magnitudes < detection_limits[band_name].to_value(u.ABmag)
        visible_fraction = np.mean(visible)
        visible_rate = visible_fraction * all_sky_rate

        visible_rates[band_name] = visible_rate

        print(
            f"{band_name}: {visible_rate:.2f} "
            f"({visible_fraction:.1%} of events visible)"
        )

    # Plot all-sky visible rates.
    band_names = list(visible_rates)
    rates = [visible_rates[band].to_value(1 / u.yr) for band in band_names]

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.bar(band_names, rates)

    ax.set_yscale("log")
    ax.set_ylabel(r"All-sky rate [yr$^{-1}$]")
    ax.set_title("Peak-visible LFBOT rate")

    fig.tight_layout()
    plt.show()


References
-----------

.. footbibliography::
