.. _transients_kilonovae:

Kilonovae
==========

Kilonovae are modeled here as the radioactively-heated, neutron-rich ejecta of a compact-object
merger (a binary neutron star or neutron star-black hole merger), calibrated to the one
well-sampled event to date, AT2017gfo (GW170817) :footcite:p:`cowperthwaite2017, waxman2018`.

This population is implemented by :class:`~uvex_transients.transients.kilonovae.Kilonova`, pairing
:class:`~uvex_transients.models.kilonovae.kne.KilonovaCoolingBlackbodySED` with the rate/duration
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
     - :math:`110^{+192}_{-82}\ \mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}` (constant in :math:`z`)
     - :footcite:t:`fishbach2026`
     - This is the *total* BNS merger rate reported by :footcite:t:`fishbach2026`, not their
       narrower GW170817-like (:math:`\sim 1.3\,M_\odot + 1.3\,M_\odot`) sub-rate. Adopting the
       total rate here means every BNS merger is assumed to produce a feasibly GW170817-like
       kilonova -- a simplifying assumption made for this simulation, not one asserted by
       :footcite:t:`fishbach2026` itself. No redshift evolution is assumed.
   * - Redshift limit
     - :math:`z = 0.19`
     - --
     - Redshift beyond which, with 95% confidence, at most 1% of events drawn from the SED's
       priors are brighter than 24.5 AB (the UVEX 1 Dwell limit) in either UVEX band, rounded up.
       See the observability summary below.
   * - Duration
     - 30 days
     - --
     - Conservative: the early, blue component this SED targets fades below detectability by
       :math:`\sim 10` days, but 30 days is used to safely bound the full light curve.

SED Model
----------

:class:`~uvex_transients.models.kilonovae.kne.KilonovaCoolingBlackbodySED` models the kilonova as a
cooling blackbody photosphere: a Gaussian-rise/broken-power-law-decline bolometric light curve
(:class:`~uvex_transients.models.lightcurves.generic.GaussianRiseBrokenPowerLawLightcurve`) times
a normalized blackbody spectral shape
(:class:`~uvex_transients.models.spectra.thermal.BlackbodySpectrum`) evaluated at a photospheric
temperature that itself declines as a power law from an early value :math:`T_0` down to a late-time
floor :math:`T_\mathrm{floor}`. The functional forms are

.. math::

    L_\mathrm{bol}(t) = L_0 \times
    \begin{cases}
        \exp\left[-\dfrac{(t-t_\mathrm{peak})^2}{2(t_\mathrm{peak}/5)^2}\right], & t \le t_\mathrm{peak} \\[6pt]
        (t/t_\mathrm{peak})^{-\alpha_1}, & t_\mathrm{peak} < t \le t_\mathrm{break} \\[6pt]
        (t_\mathrm{break}/t_\mathrm{peak})^{\alpha_2-\alpha_1}(t/t_\mathrm{peak})^{-\alpha_2}, & t > t_\mathrm{break}
    \end{cases}

.. math::

    T(t) = T_\mathrm{floor} + (T_0 - T_\mathrm{floor})\left(1 + \frac{t}{t_\mathrm{peak}}\right)^{-\alpha_T}.

This is a deliberately phenomenological choice: the rise of AT2017gfo was never actually observed
(hence the Gaussian rise is unconstrained by data and merely provides a smooth turn-on), but the
broken-power-law decline in bolometric luminosity and the power-law-to-floor cooling in
temperature both broadly track the behavior reported by :footcite:t:`waxman2018`. The luminosity
normalization is anchored to the :footcite:t:`cowperthwaite2017` measurement at 0.6 days, while the
temperature priors are set so that the median cooling curve runs through the photospheric
temperatures of :footcite:t:`waxman2018`.

.. list-table::
   :header-rows: 1
   :widths: 16 12 40 32

   * - Parameter
     - Symbol
     - Prior
     - Notes / Source
   * - ``amplitude``
     - :math:`L_0`
     - LogNormal(:math:`\log_{10}(L_0/\mathrm{erg\,s^{-1}})`; mean=41.8, :math:`\sigma`\=0.1)
     - Anchored to :footcite:t:`cowperthwaite2017`'s :math:`L\approx6.8\times10^{41}` erg/s at 0.6 d.
   * - ``t_peak``
     - :math:`t_\mathrm{peak}`
     - LogNormal(:math:`\log_{10}(t_\mathrm{peak}/0.6\,\mathrm{d})`; mean=0, :math:`\sigma`\=0.3)
     - Anchored near the 0.6 d normalization epoch, with broad scatter since the rise was unobserved.
   * - ``decline_index_1``
     - :math:`\alpha_1`
     - Uniform(0.8, 1.2)
     - Early-time post-peak decline index, :math:`L_\mathrm{bol}\sim t^{-\alpha_1}` :footcite:p:`waxman2018`.
   * - ``decline_index_2``
     - :math:`\alpha_2`
     - Uniform(3.0, 4.0)
     - Late-time post-peak decline index, :math:`L_\mathrm{bol}\sim t^{-\alpha_2}` :footcite:p:`waxman2018`.
   * - ``t_break``
     - :math:`t_\mathrm{break}`
     - Uniform(5 d, 10 d)
     - Time the decline steepens from :math:`\alpha_1` to :math:`\alpha_2` :footcite:p:`waxman2018`.
   * - ``T0``
     - :math:`T_0`
     - LogNormal(:math:`\log_{10}(T_0/\mathrm{K})`; mean=4.2, :math:`\sigma`\=0.1)
     - :math:`\approx16000` K. Median cooling curve centred on the :footcite:t:`waxman2018` temperatures.
   * - ``T_floor``
     - :math:`T_\mathrm{floor}`
     - LogNormal(:math:`\log_{10}(T_\mathrm{floor}/\mathrm{K})`; mean=3.2, :math:`\sigma`\=0.08)
     - :math:`\approx1600` K asymptotic floor, near the coolest temperatures of :footcite:p:`waxman2018`.
   * - ``alpha_T``
     - :math:`\alpha_T`
     - Uniform(0.8, 1.3)
     - Cooling index, :math:`T-T_\mathrm{floor}\sim t^{-\alpha_T}`, centred on the best fit to :footcite:p:`waxman2018`.

Simulated Light Curves
~~~~~~~~~~~~~~~~~~~~~~~

The plot below draws 1000 random parameter realizations from the priors above and shows the
resulting bolometric light curves and photospheric temperatures, together with the AT2017gfo
measurements from :footcite:t:`cowperthwaite2017` and :footcite:t:`waxman2018` that the priors
were anchored to.

.. plot::
   :include-source: false

   import numpy as np
   import matplotlib.pyplot as plt
   from astropy import units as u

   from uvex_transients.models.kilonovae import KilonovaCoolingBlackbodySED as SEDClass
   from uvex_transients.utils.lightcurve_archive import LightcurveArchive

   rng = np.random.default_rng(20260910)
   n_samples = 1000

   params = SEDClass().sample_parameters(size=n_samples, rng=rng)
   params_grid = {name: value[:, None] for name, value in params.items()}

   t = np.geomspace(0.02, 30, 200) * u.day
   L_bol = SEDClass.eval_bolometric(t, **params_grid)
   T = SEDClass.temperature(t, **params_grid)

   archive = LightcurveArchive()
   lbol_cowperthwaite = archive.table("kilonovae", "gw170817_cowperthwaite", "L_bol")
   lbol_waxman = archive.table("kilonovae", "gw170817_waxman", "L_bol")
   tphot_waxman = archive.table("kilonovae", "gw170817_waxman", "T_phot")

   fig, (ax_L, ax_T) = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True)

   for row in range(n_samples):
       ax_L.plot(t.to_value(u.day), L_bol[row].to_value(u.erg / u.s), color="C0", lw=0.4, alpha=0.06)
       ax_T.plot(t.to_value(u.day), T[row].to_value(u.K), color="C3", lw=0.4, alpha=0.06)

   ax_L.scatter(
       lbol_cowperthwaite["time"].to_value(u.day), lbol_cowperthwaite["L_bol"].to_value(u.erg / u.s),
       marker="o", s=28, color="k", edgecolor="white", linewidth=0.5, zorder=5,
       label="Cowperthwaite+2017",
   )
   ax_L.scatter(
       lbol_waxman["time"].to_value(u.day), lbol_waxman["L_bol"].to_value(u.erg / u.s),
       marker="s", s=28, color="firebrick", edgecolor="white", linewidth=0.5, zorder=5,
       label="Waxman+2018",
   )

   ax_T.scatter(
       tphot_waxman["time"].to_value(u.day), tphot_waxman["T_phot"].to_value(u.K),
       marker="s", s=28, color="firebrick", edgecolor="white", linewidth=0.5, zorder=5,
       label="Waxman+2018",
   )

   ax_L.set_xscale("log")
   ax_L.set_yscale("log")
   ax_L.set_ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
   ax_L.set_title("Kilonova: simulated bolometric light curves (n=1000)")
   ax_L.legend(loc="upper right", fontsize=8, frameon=False)

   ax_T.set_yscale("log")
   ax_T.set_xlabel("Time since merger [days]")
   ax_T.set_ylabel("Photospheric temperature [K]")
   ax_T.legend(loc="upper right", fontsize=8, frameon=False)

   fig.tight_layout()


----

Observability Summary
----------------------

The plots below show how the redshift limit follows from the SED. We draw 1000 sets of parameters
from the priors above and evaluate each of them across a grid of redshifts, in the UVEX FUV and NUV
bands, over the whole rest-frame light curve. The left panel shows each draw's peak apparent AB
magnitude, in whichever band is brighter, against redshift: a draw is detectable out to the redshift
where its line crosses the UVEX 1 Dwell limit of :math:`m<24.5`. The right panel shows the redshift
beyond which, with 95% confidence, at most 1% of the population is detectable, as a function of the
magnitude limit. The redshift limit adopted above, :math:`z=0.19`, is its value at :math:`m=24.5`,
rounded up and derived on a finer grid than this plot's.

The limit bounds the fraction of the *population* that is detectable beyond it, not the fraction of
detected events, and it does not include Milky Way extinction or sky position.

.. plot::
   :include-source: false

   import numpy as np
   import matplotlib.pyplot as plt

   from uvex_transients.missions import uvex_fast as uvex
   from uvex_transients.transients.kilonovae import Kilonova
   from uvex_transients.utils.plotting import plot_detection_horizon

   transient = Kilonova()
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

The anticipated rate of kilonovae detectable by UVEX at these limits is as follows assuming that
any event above the :math:`m<24.5` limit is detectable, and that the population is isotropic and homogeneous
in comoving volume out to :math:`z=0.19`:

.. plot::
   :include-source: false

    import numpy as np
    import matplotlib.pyplot as plt
    from astropy import units as u

    from uvex_transients.missions import uvex_fast as uvex
    from uvex_transients.transients.kilonovae import Kilonova
    from uvex_transients.utils.plotting import add_funnel_legend, get_band_color, plot_rate_bars


    rng = np.random.default_rng(20260911)
    n_samples = 3000

    # Sample the kilonova population.
    kilonova = Kilonova()
    redshift = kilonova.sample_event_redshift(n_samples, rng=rng)
    params = kilonova.sed.sample_parameters(size=n_samples, rng=rng)

    # The all-sky rate, with no survey footprint applied.
    all_sky_rate = kilonova.all_sky_rate

    # Observed-frame time corresponding to the rest-frame peak.
    t_peak_obs = params["t_peak"] * (1 + redshift)

    detection_limits = {
        "FUV": 24.5 * u.ABmag,
        "NUV": 24.5 * u.ABmag,
    }

    # Compute the number of the n_samples draws visible in each band.
    visible_counts = {}

    for band_name, bandpass in uvex.detector.bandpasses.items():
        magnitudes = kilonova.sed.mag_bandpass(
            bandpass,
            t_peak_obs,
            redshift=redshift,
            **params,
        ).to_value(u.ABmag)

        visible = magnitudes < detection_limits[band_name].to_value(u.ABmag)
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
        rate_ci=kilonova.RATE_CI,
        color=[get_band_color(band) for band in band_names],
    )

    ax.set_yscale("log")
    ax.set_ylabel(r"All-sky rate [yr$^{-1}$]")
    ax.set_title("Peak-visible kilonova rate")
    add_funnel_legend(ax, loc="lower right")

    fig.tight_layout()
    plt.show()


References
-----------

.. footbibliography::
