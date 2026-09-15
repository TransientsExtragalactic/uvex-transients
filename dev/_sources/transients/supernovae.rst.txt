.. _transients_supernovae:

Type IIP Supernovae
=====================

Type IIP core-collapse supernovae (CCSNe) are the explosions of hydrogen-rich massive stars whose
extended envelopes produce a characteristic weeks-to-months-long luminosity "plateau" as a
recombination front recedes through the ejecta. Two populations are modeled here: ordinary Type IIP
SNe, and a subset that shows a distinct early-time (first few days) luminosity excess -- attributed
to shock breakout through and/or collisional heating of confined circumstellar material shed by the
progenitor shortly before explosion -- typified by SN 2023ixf :footcite:p:`hsu2025` and SN 2024ggi
:footcite:p:`chen2024`, both of which show a fast early rise to a hot (:math:`\sim2.5\times10^4` K)
peak within the first few days. UVEX's UV sensitivity makes it well suited to catching precisely
this brief, hot early phase, which fades quickly out of optical bands.

Both populations are implemented by :class:`~uvex_transients.transients.supernovae.TypeIIPSNe` and
:class:`~uvex_transients.transients.supernovae.TypeIIPExcessSNe`, pairing
:class:`~uvex_transients.models.supernovae.IIp_excess.TypeIIPSED` and
:class:`~uvex_transients.models.supernovae.IIp_excess.TypeIIPExcessSED` respectively with the
rate/duration metadata described below.

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
     - :math:`R_\mathrm{CC}(z) = k h^2 \psi_\mathrm{UV}(z)`; Type IIP 40%, IIP+excess 12% of
       :math:`R_\mathrm{CC}(z)`
     - :footcite:t:`strolger2015`, :footcite:t:`madau2014`, :footcite:t:`li2011`
     - Both populations are fixed fractions of a single, shared total CCSNe rate, where
       :math:`\psi_\mathrm{UV}(z)` is the UV-derived cosmic star-formation-rate density
       :footcite:p:`madau2014` and :math:`k` is fit directly to CCSNe counts by
       :footcite:t:`strolger2015` (CANDELS/CLASH HST supernova surveys, out to :math:`z\approx2.5`)
       -- tying the rate to :math:`R_\mathrm{CC}(z)` lets it evolve self-consistently with cosmic
       star formation. Ordinary Type IIP SNe make up 40% of :math:`R_\mathrm{CC}(z)`, the local
       Type IIP fraction of the core-collapse population from the volume-limited Lick Observatory
       Supernova Search sample :footcite:p:`li2011`; of those, roughly 30% show the SN
       2023ixf/SN 2024ggi-like early excess, so that subclass is assigned
       :math:`0.40\times0.30=12\%` of :math:`R_\mathrm{CC}(z)`.
   * - Redshift limit
     - :math:`z = 1` (Type IIP), :math:`z = 2` (Type IIP + excess)
     - --
     - The early-interacting population's higher peak temperatures and correspondingly bluer,
       more UV-detectable emission justify its more generous limit.
   * - Duration
     - 200 days (both populations)
     - --
     - Generous enough to cover the plateau and the transition to the (unmodeled) nebular phase.

SED Model
----------

Type IIP
~~~~~~~~~

:class:`~uvex_transients.models.supernovae.IIp_excess.TypeIIPSED` pairs the same rise/plateau/decline
bolometric light curve introduced by :footcite:t:`2019ApJ...884...83V` for photometric SN
classification (:class:`~uvex_transients.models.lightcurves.generic.VillarLightcurve`) with a
smooth broken-power-law-with-floor photospheric temperature, both referenced to a shared time
:math:`t_0`:

.. math::

    L_\mathrm{bol}(t) = A \times
    \begin{cases}
        \dfrac{1+\beta(t-t_0)}{1+\exp[-(t-t_0)/\tau_r]}, & t < t_1, \\[8pt]
        \dfrac{[1+\beta(t_1-t_0)]\exp[-(t-t_1)/\tau_f]}{1+\exp[-(t-t_0)/\tau_r]}, & t \ge t_1,
    \end{cases}
    \qquad
    T(t) = T_f + (T_\mathrm{peak}-T_f)\left[(t/t_0)^{-s\alpha_r} + (t/t_0)^{s\alpha_d}\right]^{-1/s}.

Because this codebase does not yet have a sample of UV-selected Type IIP light curves on which to
fit subtype-specific priors, several of the temperature-law shape parameters
(``beta``, ``tau_rise``, ``tau_fall``, ``alpha_rise``, ``alpha_decay``, ``smoothing``) are held
fixed at representative values rather than sampled, leaving only the overall amplitude, plateau
timing, and peak/floor temperatures free.

.. list-table::
   :header-rows: 1
   :widths: 16 12 40 32

   * - Parameter
     - Symbol
     - Prior
     - Notes / Source
   * - ``amplitude``
     - :math:`A`
     - Normal(:math:`\log_{10}(A/\mathrm{erg\,s^{-1}})`; mean=42, :math:`\sigma`\=0.1)
     - Luminosity normalization.
   * - ``t0``
     - :math:`t_0`
     - Uniform(2 d, 5 d)
     - Peak/reference time, shared by the light curve and the temperature law.
   * - ``t1``
     - :math:`t_1`
     - Uniform(50 d, 120 d)
     - Plateau end time :footcite:p:`2019ApJ...884...83V`.
   * - ``beta``
     - :math:`\beta`
     - Fixed (0/day)
     - Plateau slope.
   * - ``tau_rise``
     - :math:`\tau_r`
     - Fixed (1 day)
     - Logistic rise timescale.
   * - ``tau_fall``
     - :math:`\tau_f`
     - Fixed (50 day)
     - Post-plateau exponential decline timescale.
   * - ``T_peak``
     - :math:`T_\mathrm{peak}`
     - Normal(:math:`\log_{10}(T_\mathrm{peak}/\mathrm{K})`; mean=4.0, :math:`\sigma`\=0.15)
     - Approximate peak photospheric temperature.
   * - ``T_floor``
     - :math:`T_f`
     - Normal(:math:`\log_{10}(T_f/\mathrm{K})`; mean=3.8, :math:`\sigma`\=0.08)
     - Asymptotic (recombination) photospheric temperature floor.
   * - ``alpha_rise``, ``alpha_decay``, ``smoothing``
     - :math:`\alpha_r,\ \alpha_d,\ s`
     - Fixed (0.8, 1, 4)
     - Temperature-law shape parameters.

Type IIP + Early-Time Excess
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~uvex_transients.models.supernovae.IIp_excess.TypeIIPExcessSED` adds a sigmoid-shaped early-time
excess on top of the same :class:`~uvex_transients.models.supernovae.IIp_excess.TypeIIPSED` light curve and
(unchanged) temperature law:

.. math::

    L_\mathrm{bol}(t) = L_\mathrm{Villar}(t) + L_\mathrm{excess}(t), \qquad
    L_\mathrm{excess}(t) = A_e
    \left[1-\exp\!\left(-\frac{t-t_0}{\tau_r}\right)\right]
    \exp\!\left(-\frac{t-t_0}{\tau_e}\right)\ \ (t\ge t_0),

sharing the same rise timescale :math:`\tau_r` and reference time :math:`t_0` as the Villar
component. The excess amplitude prior is set from the two best-observed examples of this subclass:
SN 2024ggi peaked at :math:`\sim1.5\times10^{43}` erg/s :footcite:p:`chen2024` and SN 2023ixf at
:math:`\sim4\times10^{43}` erg/s :footcite:p:`hsu2025`. As with the base model, the underlying
sample is too small to fit reliable priors from, so the excess decay timescale is held fixed.

.. list-table::
   :header-rows: 1
   :widths: 16 12 40 32

   * - Parameter
     - Symbol
     - Prior
     - Notes / Source
   * - ``excess_amplitude``
     - :math:`A_e`
     - Normal(:math:`\log_{10}(A_e/\mathrm{erg\,s^{-1}})`; mean=43.5, :math:`\sigma`\=0.2)
     - Early-excess luminosity normalization; SN 2024ggi/SN 2023ixf peaked at
       :math:`\sim1.5\text{-}4\times10^{43}` erg/s :footcite:p:`chen2024,hsu2025`.
   * - ``tau_excess``
     - :math:`\tau_e`
     - Normal(:math:`\tau_e`/6 day; mean=1.0, :math:`\sigma`\=0.1)
     - Excess decay timescale.

All other parameters (``amplitude``, ``t0``, ``t1``, ``T_peak``, ``T_floor``, and the fixed shape
parameters) are inherited unchanged from :class:`~uvex_transients.models.supernovae.IIp_excess.TypeIIPSED`
above.

Simulated Light Curves
~~~~~~~~~~~~~~~~~~~~~~~

The plots below each draw 1000 random parameter realizations from the corresponding model's
priors and show the resulting bolometric light curves and photospheric temperatures -- first for
the ordinary Type IIP model, then for the Type IIP + early-excess variant. Each is overlaid with
the observed bolometric light curves and photospheric temperatures of two well-observed examples
of the relevant subclass: SN 1999em and SN 2003hn :footcite:p:`bersten2009` for the ordinary
population, and SN 2023ixf :footcite:p:`hsu2025` and SN 2024ggi :footcite:p:`chen2024` for the
early-excess population.

.. plot::
   :include-source: false

   from pathlib import Path

   import numpy as np
   import matplotlib.pyplot as plt
   from astropy import units as u
   from astropy.table import Table

   import uvex_transients
   from uvex_transients.models.supernovae import TypeIIPSED as SEDClass

   rng = np.random.default_rng(20260910)
   n_samples = 1000

   params = SEDClass().sample_parameters(size=n_samples, rng=rng)
   params_grid = {name: value[:, None] for name, value in params.items()}

   t = np.geomspace(0.1, 200, 200) * u.day
   L_bol = SEDClass.eval_bolometric(t, **params_grid)
   T = SEDClass.temperature(t, **params_grid)

   fig, (ax_L, ax_T) = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True)

   for row in range(n_samples):
       ax_L.plot(t.to_value(u.day), L_bol[row].to_value(u.erg / u.s), color="C0", lw=0.4, alpha=0.06)
       ax_T.plot(t.to_value(u.day), T[row].to_value(u.K), color="C3", lw=0.4, alpha=0.06)

   data_dir = Path(uvex_transients.__file__).parent.parent / "test_data" / "transients"
   observed_iip_sne = [
       ("1999em_bersten2009.txt", "SN 1999em (Bersten & Hamuy 2009)", "o", "k"),
       ("2003hn_bersten2009.txt", "SN 2003hn (Bersten & Hamuy 2009)", "s", "firebrick"),
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

   ax_L.set_xscale("log")
   ax_L.set_yscale("log")
   ax_L.set_ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
   ax_L.set_title("Type IIP SNe: simulated bolometric light curves (n=1000)")
   ax_L.legend(loc="upper right", fontsize=8, frameon=False)

   ax_T.set_yscale("log")
   ax_T.set_xlabel("Time since explosion [days]")
   ax_T.set_ylabel("Photospheric temperature [K]")

   fig.tight_layout()

.. plot::
   :include-source: false

   from pathlib import Path

   import numpy as np
   import matplotlib.pyplot as plt
   from astropy import units as u
   from astropy.table import Table

   import uvex_transients
   from uvex_transients.models.supernovae import TypeIIPExcessSED as SEDClass

   rng = np.random.default_rng(20260910)
   n_samples = 1000

   params = SEDClass().sample_parameters(size=n_samples, rng=rng)
   params_grid = {name: value[:, None] for name, value in params.items()}

   t = np.geomspace(0.1, 200, 200) * u.day
   L_bol = SEDClass.eval_bolometric(t, **params_grid)
   T = SEDClass.temperature(t, **params_grid)

   fig, (ax_L, ax_T) = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True)

   for row in range(n_samples):
       ax_L.plot(t.to_value(u.day), L_bol[row].to_value(u.erg / u.s), color="C0", lw=0.4, alpha=0.06)
       ax_T.plot(t.to_value(u.day), T[row].to_value(u.K), color="C3", lw=0.4, alpha=0.06)

   data_dir = Path(uvex_transients.__file__).parent.parent / "test_data" / "transients"
   observed_excess_sne = [
       ("2023ixf_hsu2025.txt", "SN 2023ixf (Hsu+2025)", "o", "k"),
       ("2024ggi_chen2024.txt", "SN 2024ggi (Chen+2024)", "s", "firebrick"),
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
   ax_L.set_title("Type IIP + early excess SNe: simulated bolometric light curves (n=1000)")
   ax_L.legend(loc="upper right", fontsize=8, frameon=False)

   ax_T.set_yscale("log")
   ax_T.set_xlabel("Time since explosion [days]")
   ax_T.set_ylabel("Photospheric temperature [K]")

   fig.tight_layout()


----

Observability Summary
----------------------

Below are the redshifts :math:`z` and corresponding bandpass calculated peak apparent AB magnitudes
:math:`m_\mathrm{AB}` of 3000 simulated events from each population drawn from the priors above,
with the UVEX 1 Dwell limit of :math:`m<24.5` overplotted -- first for the ordinary Type IIP
population, then for the Type IIP + early-excess population. Because neither model's bolometric
light curve peaks exactly at a single named parameter (the excess component, in particular, peaks
some time *after* its own reference time ``t0``, not at it), the peak apparent magnitude is found
by a numerical search over each event's light curve rather than evaluated at a fixed time. Findings
here justify our confidence in the :math:`z=1` and :math:`z=2` redshift limits adopted for the two
populations, respectively.

.. plot::
   :include-source: false

    import numpy as np
    import matplotlib.pyplot as plt
    from astropy import units as u
    from scipy.stats import gaussian_kde

    from m4opt.missions import uvex
    from uvex_transients.transients.supernovae import TypeIIPSNe
    from uvex_transients.models.supernovae import TypeIIPSED as SEDClass

    rng = np.random.default_rng(20260911)
    n_samples = 3000

    sn = TypeIIPSNe()
    z = sn.sample_event_redshift(n_samples, rng=rng)
    params = SEDClass().sample_parameters(size=n_samples, rng=rng)
    params_grid = {name: value[:, None] for name, value in params.items()}

    # Numerically search each event's own light curve for its brightest (peak) apparent
    # magnitude, rather than assuming a single named parameter marks the true peak.
    t_grid_rest = np.geomspace(0.1, 200, 200) * u.day
    t_obs_grid = t_grid_rest[None, :] * (1.0 + z)[:, None]
    z_grid_bcast = np.broadcast_to(z[:, None], t_obs_grid.shape)

    bandpasses = uvex.detector.bandpasses
    band_names = list(bandpasses)

    fig, axes = plt.subplots(1, len(band_names), figsize=(10.5, 4.8), sharey=True)

    for ax, band_name in zip(axes, band_names):
        mag_curve = SEDClass.mag_bandpass(
            bandpasses[band_name], t_obs_grid, redshift=z_grid_bcast, **params_grid
        ).to_value(u.ABmag)
        mag = np.nanmin(mag_curve, axis=1)
        finite = np.isfinite(mag)
        z_finite, mag_finite = z[finite], mag[finite]

        ax.scatter(z_finite, mag_finite, s=5, ec='k',fc='k',alpha=0.5, label="Simulated events")

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
        ax.set_ylim([55, 15])

    axes[0].set_ylabel("Peak apparent AB magnitude")
    fig.suptitle(f"Type IIP SNe: peak apparent magnitude vs. redshift (n={n_samples})")
    fig.tight_layout()
    plt.show()

.. plot::
   :include-source: false

    import numpy as np
    import matplotlib.pyplot as plt
    from astropy import units as u
    from scipy.stats import gaussian_kde

    from m4opt.missions import uvex
    from uvex_transients.transients.supernovae import TypeIIPExcessSNe
    from uvex_transients.models.supernovae import TypeIIPExcessSED as SEDClass

    rng = np.random.default_rng(20260911)
    n_samples = 3000

    sn = TypeIIPExcessSNe()
    z = sn.sample_event_redshift(n_samples, rng=rng)
    params = SEDClass().sample_parameters(size=n_samples, rng=rng)
    params_grid = {name: value[:, None] for name, value in params.items()}

    # Numerically search each event's own light curve for its brightest (peak) apparent
    # magnitude -- the early-excess component itself is exactly zero at t0 and peaks a
    # little afterwards, so no single named parameter marks the true peak here.
    t_grid_rest = np.geomspace(0.1, 200, 200) * u.day
    t_obs_grid = t_grid_rest[None, :] * (1.0 + z)[:, None]
    z_grid_bcast = np.broadcast_to(z[:, None], t_obs_grid.shape)

    bandpasses = uvex.detector.bandpasses
    band_names = list(bandpasses)

    fig, axes = plt.subplots(1, len(band_names), figsize=(10.5, 4.8), sharey=True)

    for ax, band_name in zip(axes, band_names):
        mag_curve = SEDClass.mag_bandpass(
            bandpasses[band_name], t_obs_grid, redshift=z_grid_bcast, **params_grid
        ).to_value(u.ABmag)
        mag = np.nanmin(mag_curve, axis=1)
        finite = np.isfinite(mag)
        z_finite, mag_finite = z[finite], mag[finite]

        ax.scatter(z_finite, mag_finite, s=5, ec='k',fc='k',alpha=0.5, label="Simulated events")

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
        ax.set_ylim([60, 15])

    axes[0].set_ylabel("Peak apparent AB magnitude")
    fig.suptitle(f"Type IIP + early excess SNe: peak apparent magnitude vs. redshift (n={n_samples})")
    fig.tight_layout()
    plt.show()

The anticipated rate of each population detectable by UVEX at these limits is as follows assuming
that any event above the :math:`m<24.5` limit is detectable, and that each population is isotropic
and homogeneous in comoving volume out to its own redshift limit:

.. plot::
   :include-source: false

    import numpy as np
    import matplotlib.pyplot as plt
    from astropy import units as u

    from m4opt.missions import uvex
    from uvex_transients.transients.supernovae import TypeIIPSNe, TypeIIPExcessSNe


    rng = np.random.default_rng(20260911)
    n_samples = 3000

    detection_limits = {
        "FUV": 24.5 * u.ABmag,
        "NUV": 24.5 * u.ABmag,
    }

    populations = {
        "Type IIP": TypeIIPSNe(),
        "Type IIP + excess": TypeIIPExcessSNe(),
    }

    # Compute peak-visible rates for each population.
    visible_rates = {band_name: [] for band_name in uvex.detector.bandpasses}
    population_names = list(populations)

    for name, sn in populations.items():
        redshift = sn.sample_event_redshift(n_samples, rng=rng)
        params = sn.sed.sample_parameters(size=n_samples, rng=rng)
        params_grid = {pname: value[:, None] for pname, value in params.items()}

        # Convert the integrated rate per steradian to an all-sky rate.
        all_sky_rate = 4 * np.pi * sn.integrated_event_rate * u.sr

        # Numerically search each event's own light curve for its brightest (peak)
        # apparent magnitude -- see the discussion above for why neither model's peak
        # sits at a single named parameter.
        t_grid_rest = np.geomspace(0.1, 200, 200) * u.day
        t_obs_grid = t_grid_rest[None, :] * (1.0 + redshift)[:, None]
        z_grid_bcast = np.broadcast_to(redshift[:, None], t_obs_grid.shape)

        for band_name, bandpass in uvex.detector.bandpasses.items():
            mag_curve = sn.sed.mag_bandpass(
                bandpass,
                t_obs_grid,
                redshift=z_grid_bcast,
                **params_grid,
            ).to_value(u.ABmag)
            magnitudes = np.nanmin(mag_curve, axis=1)

            visible = magnitudes < detection_limits[band_name].to_value(u.ABmag)
            visible_fraction = np.mean(visible)
            visible_rate = visible_fraction * all_sky_rate

            visible_rates[band_name].append(visible_rate.to_value(1 / u.yr))

            print(
                f"{name} {band_name}: {visible_rate:.2f} "
                f"({visible_fraction:.1%} of events visible)"
            )

    # Plot all-sky visible rates, grouped by band.
    band_names = list(visible_rates)
    x = np.arrange(len(population_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))

    for offset, band_name in zip((-width / 2, width / 2), band_names):
        ax.bar(x + offset, visible_rates[band_name], width, label=band_name)

    ax.set_xticks(x)
    ax.set_xticklabels(population_names)
    ax.set_yscale("log")
    ax.set_ylabel(r"All-sky rate [yr$^{-1}$]")
    ax.set_title("Peak-visible Type IIP SNe rate")
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    fig.tight_layout()
    plt.show()


References
-----------

.. footbibliography::
