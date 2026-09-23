.. _transients_tdes:

Tidal Disruption Events
=========================

A tidal disruption event (TDE) occurs when a star passes close enough to a (typically
super-massive) black hole that the hole's tidal field exceeds the star's self-gravity, unbinding
and disrupting it; roughly half of the stellar debris remains bound and eventually falls back onto
the black hole, powering a luminous, months-long flare. Optically/UV-selected TDEs, the class
UVEX is sensitive to, are observed to radiate as blue, roughly constant-temperature thermal
sources with characteristic blackbody temperatures of a few :math:`\times10^4` K
:footcite:p:`2021ApJ...908....4V`, and this SED's default priors are built from the empirical
rise/decline/temperature statistics of that work's homogeneously-analyzed sample of 39 optical/UV
TDEs.

This population is implemented by :class:`~uvex_transients.transients.TDEs.TidalDisruptionEvent`,
pairing :class:`~uvex_transients.models.tdes.van_velzen.VanVelzenTDESED` with the rate/duration
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
     - :math:`3.1\times10^{-7}\ \mathrm{Mpc^{-3}\,yr^{-1}}` (constant in :math:`z`)
     - :footcite:t:`yao2023`
     - Maximum-volume-corrected demographic rate from 33 spectroscopically-confirmed TDEs from
       three years of the Zwicky Transient Facility. Taken as constant in :math:`z`, since its
       evolution remains actively debated -- :footcite:t:`karmen2026` show the observed
       redshift-dependent TDE rate is highly sensitive to the poorly-constrained evolution of the
       supermassive black hole mass function itself.
   * - Redshift limit
     - :math:`z = 2`
     - --
     - Generous relative to the timescales and luminosities of the observed optical/UV TDE
       population.
   * - Duration
     - 200 days
     - --
     - Generous relative to the timescales and luminosities of the observed optical/UV TDE
       population.

SED Model
----------

There are two TDE SED models which are available:

- :class:`~uvex_transients.models.tdes.van_velzen.VanVelzenTDESED` implements the
  Gaussian-rise/exponential-decline parameterization of :footcite:t:`2021ApJ...908....4V`, with a
  constant-temperature blackbody photosphere. This is the simplest model that reproduces the
  observed optical/UV TDE population.
- :class:`~uvex_transients.models.tdes.alush_stone.AlushStoneTDESED` implements a more complex model
  that includes a late-time component, representing a magnetized accretion disk.

By default, the transient class :class:`~uvex_transients.transients.TDEs.TidalDisruptionEvent`
uses the former, simpler model, a constant-temperature blackbody modulated by a Gaussian-rise,
exponential-decay light curve:

.. math::

    L_\nu(\nu, t) = L_0 \cdot \ell(t) \cdot \frac{\pi B_\nu(\nu, T)}{\sigma_\mathrm{SB} T^4},
    \qquad
    \ell(t) = \begin{cases}
        \exp\left[-\dfrac{(t-t_\mathrm{peak})^2}{2\sigma^2}\right] & t < t_\mathrm{peak} \\[6pt]
        \exp\left[-\dfrac{t-t_\mathrm{peak}}{\tau}\right] & t \ge t_\mathrm{peak}
        \end{cases},

with :math:`t_\mathrm{peak}=5\sigma`. This is the simplest model that reproduces the observed
optical/UV TDE population and is sufficient for most population-level UVEX forecasting, where the
late-time disk plateau discussed below is faint and rarely detected.

.. list-table::
   :header-rows: 1
   :widths: 16 12 40 32

   * - Parameter
     - Symbol
     - Prior
     - Notes / Source
   * - ``amplitude``
     - :math:`L_0`
     - LogNormal(:math:`\log_{10}(L_0/\mathrm{erg\,s^{-1}})`; mean=43.8, :math:`\sigma`\=0.3)
     - Peak bolometric luminosity, :math:`L_0=L_\mathrm{bol}(t_\mathrm{peak})`
       :footcite:p:`2021ApJ...908....4V`.
   * - ``temperature``
     - :math:`T`
     - LogNormal(:math:`\log_{10}(T/\mathrm{K})`; mean=4.3, :math:`\sigma`\=0.1)
     - Photospheric temperature, :math:`\approx2\times10^4` K :footcite:p:`2021ApJ...908....4V`.
   * - ``sigma_rise``
     - :math:`\sigma`
     - LogNormal(:math:`\log_{10}(\sigma/\mathrm{d})`; mean=1.3, :math:`\sigma`\=0.3)
     - Gaussian width of the pre-peak rise :footcite:p:`2021ApJ...908....4V`.
   * - ``tau_decline``
     - :math:`\tau`
     - LogNormal(:math:`\log_{10}(\tau/\mathrm{d})`; mean=2, :math:`\sigma`\=0.2)
     - Exponential decline timescale after peak :footcite:p:`2021ApJ...908....4V`.

Plateau visibility: AlushStoneTDESED
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~uvex_transients.models.tdes.alush_stone.AlushStoneTDESED` should be used instead of the
default model whenever a study specifically cares about the *visibility of the late-time disk
plateau* (e.g. forecasting how often UVEX would detect the plateau itself, or how it biases
late-time photometry), rather than population-level early-time behavior. It adds a second,
separately-normalized blackbody component on top of the same early-time photosphere used by
:class:`~uvex_transients.models.tdes.van_velzen.VanVelzenTDESED`, one that smoothly softens from a
flat plateau to a power-law decline:

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
late-time decline persisting for decades to centuries.

.. list-table::
   :header-rows: 1
   :widths: 16 12 40 32

   * - Parameter
     - Symbol
     - Prior
     - Notes / Source
   * - ``amplitude``
     - :math:`L_0`
     - LogNormal(:math:`\log_{10}(L_0/\mathrm{erg\,s^{-1}})`; mean=43.8, :math:`\sigma`\=0.3)
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

The plot below draws 1000 random parameter realizations of the default
:class:`~uvex_transients.models.tdes.van_velzen.VanVelzenTDESED` model from the priors above and
shows the resulting bolometric light curves, recentered on each realization's own peak time to
match the peak-relative observed bolometric light curves of five optical/UV TDEs from
:footcite:t:`2021ApJ...908....4V`.

.. plot::
   :include-source: false

    import numpy as np
    import matplotlib.pyplot as plt
    from astropy import units as u

    from uvex_transients.transients.TDEs import TidalDisruptionEvent
    from uvex_transients.models.lightcurves.generic import GREDLightcurve
    from uvex_transients.utils.lightcurve_archive import LightcurveArchive

    rng = np.random.default_rng(20260910)
    n_samples = 1000

    TDEs = TidalDisruptionEvent()
    params = TDEs.sed.sample_parameters(size=n_samples, rng=rng)
    params_grid = {name: value[:, None] for name, value in params.items()}

    # Recenter each realization on its own peak time (t_peak = 5 sigma_rise) so the
    # simulated curves line up with the peak-relative observed TDE data below.
    t_rel = np.linspace(-30, 200, 400) * u.day
    t_peak = 5 * params_grid["sigma_rise"]
    t = t_rel + t_peak

    L_bol = TDEs.sed.eval_bolometric(t, **params_grid)

    archive = LightcurveArchive()
    observed_tdes = [
        ("2018hyz_vanvelzen", "AT2018hyz (van Velzen+2021)", "o", "k"),
        ("2019qiz_vanvelzen", "AT2019qiz (van Velzen+2021)", "s", "firebrick"),
        ("2018lna_vanvelzen", "AT2018lna (van Velzen+2021)", "^", "darkorange"),
        ("2018iih_vanvelzen", "AT2018iih (van Velzen+2021)", "D", "seagreen"),
        ("2019mha_vanvelzen", "AT2019mha (van Velzen+2021)", "v", "mediumpurple"),
    ]

    fig, ax_L = plt.subplots(figsize=(6.4, 4.8))

    for row in range(n_samples):
        ax_L.plot(t_rel.to_value(u.day), L_bol[row].to_value(u.erg / u.s), color="C0", lw=0.4, alpha=0.06)

    for suffix, label, marker, color in observed_tdes:
        lbol_obs = archive.table("tdes", suffix, "L_bol")
        ax_L.scatter(
            lbol_obs["time"].to_value(u.day), lbol_obs["L_bol"].to_value(u.erg / u.s),
            marker=marker, s=28, color=color, edgecolor="white", linewidth=0.5, zorder=5,
            label=label,
        )

    ax_L.set_xlim(-30, 200)
    ax_L.set_yscale("log")
    ax_L.set_ylim(1e41, 1e45)
    ax_L.set_xlabel("Time since peak [days]")
    ax_L.set_ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
    ax_L.set_title("Tidal disruption events: simulated bolometric light curves (n=1000)")
    ax_L.legend(loc="upper right", fontsize=8, frameon=False)

    fig.tight_layout()
    plt.show()


----

Observability Summary
----------------------

Below are the redshifts :math:`z` and corresponding bandpass calculated peak apparent AB magnitudes
:math:`m_\mathrm{AB}` of 3000 simulated TDEs drawn from the priors above, with
the UVEX 1 Dwell limit of :math:`m<24.5` overplotted. Findings here justify our confidence in a
:math:`z=2` redshift limit for this population.

.. plot::
   :include-source: false

    import numpy as np
    import matplotlib.pyplot as plt
    from astropy import units as u
    from scipy.stats import gaussian_kde

    from m4opt.missions import uvex
    from uvex_transients.transients.TDEs import TidalDisruptionEvent

    rng = np.random.default_rng(20260911)
    n_samples = 3000

    tde = TidalDisruptionEvent()
    z = tde.sample_event_redshift(n_samples, rng=rng)
    params = tde.sed.sample_parameters(size=n_samples, rng=rng)

    # Observed-frame time of rest-frame peak (5 sigma_rise into the Gaussian rise), i.e.
    # where each event is brightest as seen by UVEX.
    t_obs_peak = 5 * params["sigma_rise"] * (1.0 + z)

    bandpasses = uvex.detector.bandpasses
    band_names = list(bandpasses)

    fig, axes = plt.subplots(1, len(band_names), figsize=(10.5, 4.8), sharey=True)

    for ax, band_name in zip(axes, band_names):
        mag = tde.sed.mag_bandpass(bandpasses[band_name], t_obs_peak, redshift=z, **params).to_value(u.ABmag)
        finite = np.isfinite(mag)
        z_finite, mag_finite = z[finite], mag[finite]

        ax.scatter(z_finite, mag_finite, s=5, ec='k', fc='k', alpha=0.5, label="Simulated events")

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
        ax.set_ylim([32, 15])

    axes[0].set_ylabel("Peak apparent AB magnitude")
    fig.suptitle(f"TDEs: peak apparent magnitude vs. redshift (n={n_samples})")
    fig.tight_layout()
    plt.show()

The anticipated rate of TDEs detectable by UVEX at these limits is as follows assuming that
any event above the :math:`m<24.5` limit is detectable, and that the population is isotropic and homogeneous
in comoving volume out to :math:`z=2`:

.. plot::
   :include-source: false

    import numpy as np
    import matplotlib.pyplot as plt
    from astropy import units as u

    from m4opt.missions import uvex
    from uvex_transients.transients.TDEs import TidalDisruptionEvent


    rng = np.random.default_rng(20260911)
    n_samples = 3000

    # Sample the TDE population.
    tde = TidalDisruptionEvent()
    redshift = tde.sample_event_redshift(n_samples, rng=rng)
    params = tde.sed.sample_parameters(size=n_samples, rng=rng)

    # The all-sky rate, with no survey footprint applied.
    all_sky_rate = tde.all_sky_rate

    # Observed-frame time corresponding to the rest-frame peak.
    t_peak_obs = 5 * params["sigma_rise"] * (1 + redshift)

    detection_limits = {
        "FUV": 24.5 * u.ABmag,
        "NUV": 24.5 * u.ABmag,
    }

    # Compute peak-visible rates.
    visible_rates = {}

    for band_name, bandpass in uvex.detector.bandpasses.items():
        magnitudes = tde.sed.mag_bandpass(
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
    ax.set_title("Peak-visible TDE rate")

    fig.tight_layout()
    plt.show()


References
-----------

.. footbibliography::
