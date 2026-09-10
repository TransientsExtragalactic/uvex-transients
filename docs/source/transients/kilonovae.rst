.. _transients_kilonovae:

Kilonovae
==========

Kilonovae are the optical/UV/IR transients produced by the radioactively-heated, neutron-rich
ejecta of a compact-object merger -- most plausibly a binary neutron star (BNS) merger, though a
neutron star-black hole merger can also produce one. They are the electromagnetic counterpart of
greatest interest to a gravitational-wave-triggered UV survey like UVEX, since they (unlike a
short gamma-ray burst) are detectable across essentially the full solid angle around the merger
rather than only along a narrow relativistic jet.

The only kilonova with a well-sampled UV/optical/IR light curve to date is AT2017gfo, the
counterpart of GW170817. Its early-time (:math:`\lesssim 1` day) optical/UV emission was
significantly bluer and hotter than its later, redder emission, a signature generally attributed
to a lower-opacity, lanthanide-poor ejecta component dominating early on before a higher-opacity,
lanthanide-rich component takes over at longer wavelengths and later times
:footcite:p:`cowperthwaite2017`. UVEX's UV bandpasses are most sensitive to precisely this
early-time blue component, which is also the piece of the light curve for which GW170817 itself
gives the best empirical anchor: :footcite:t:`cowperthwaite2017` find a blackbody temperature of
:math:`T \approx 8300` K and bolometric luminosity :math:`L_\mathrm{bol} \approx 5\times10^{41}`
erg/s at 0.6 days post-merger, and :footcite:t:`waxman2018` characterize the subsequent bolometric
decline and cooling as consistent with power laws in each of luminosity and temperature. This
population is implemented by
:class:`~uvex_transients.transients.kilonovae.Kilonova`, pairing
:class:`~uvex_transients.models.kilonovae.kne.KilonovaCoolingBlackbodySED` with the rate/duration
metadata described below.

Transient Rates
----------------

The kilonova rate is tied directly to the BNS merger rate measured from gravitational-wave
observations. :footcite:t:`fishbach2026` report a total BNS merger rate of
:math:`28`-:math:`300\ \mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}` from GWTC-4, of which
:math:`53^{+176}_{-49}\ \mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}` is attributed specifically to
GW170817-like (:math:`\sim 1.3\,M_\odot + 1.3\,M_\odot`) systems. Because not every BNS merger is
expected to produce a kilonova as luminous and as blue as AT2017gfo, the GW170817-like sub-rate is
adopted here as a conservative, directly-calibrated estimate of the rate of kilonovae with the
properties this SED models, rather than the full (and less certain) BNS rate. The rate is taken as
constant in redshift, since there is no observational handle yet on its evolution.

The redshift limit is set from a luminosity argument rather than a direct kilonova
non-detection limit: requiring a source with bolometric luminosity below
:math:`10^{43}\,\mathrm{erg/s}` -- already well in excess of anything reported for a kilonova in
the literature -- to remain above a conservative UVEX band-limiting magnitude of :math:`m<27`
gives (with no K-correction, assuming a spectrum flat across the UVEX bandpasses) a redshift limit
of :math:`z=2`. The 30-day duration window is likewise conservative: the early, blue kilonova
component this SED targets is expected to fade below detectability by :math:`\sim 10` days, but 30
days is used to safely bound the full light curve.

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - Rate
     - Redshift Limit
     - Duration
     - Source
   * - :math:`53^{+176}_{-49}\ \mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}` (constant in :math:`z`)
     - :math:`z = 2`
     - 30 days
     - :footcite:t:`fishbach2026`

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

    T(t) = T_\mathrm{floor} + (T_0 - T_\mathrm{floor})\left(1 + \frac{t}{t_\mathrm{peak}/5}\right)^{-\alpha_T}.

This is a deliberately phenomenological choice: the rise of AT2017gfo was never actually observed
(hence the Gaussian rise is unconstrained by data and merely provides a smooth turn-on), but the
broken-power-law decline in bolometric luminosity and the power-law-to-floor cooling in
temperature both broadly track the behavior reported by :footcite:t:`waxman2018`, with the
normalization of both anchored to the :footcite:t:`cowperthwaite2017` measurement at 0.6 days.

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
     - LogNormal(:math:`\log_{10}(T_0/\mathrm{K})`; mean=3.9, :math:`\sigma`\=0.1)
     - :math:`\approx7900` K, anchored to :footcite:t:`cowperthwaite2017`'s :math:`T\approx8300` K at 0.6 d.
   * - ``T_floor``
     - :math:`T_\mathrm{floor}`
     - LogNormal(:math:`\log_{10}(T_\mathrm{floor}/\mathrm{K})`; mean=3.4, :math:`\sigma`\=0.08)
     - :math:`\approx2500` K asymptotic floor :footcite:p:`waxman2018`.
   * - ``alpha_T``
     - :math:`\alpha_T`
     - Uniform(0.3, 0.7)
     - Early-time cooling index, :math:`T\sim t^{-\alpha_T}` :footcite:p:`waxman2018`.

Simulated Light Curves
~~~~~~~~~~~~~~~~~~~~~~~

The plot below draws 1000 random parameter realizations from the priors above and shows the
resulting bolometric light curves and photospheric temperatures.

.. plot::
   :include-source: false

   import numpy as np
   import matplotlib.pyplot as plt
   from astropy import units as u

   from uvex_transients.models.kilonovae import KilonovaCoolingBlackbodySED as SEDClass

   rng = np.random.default_rng(20260910)
   n_samples = 1000

   params = SEDClass().sample_parameters(size=n_samples, rng=rng)
   params_grid = {name: value[:, None] for name, value in params.items()}

   t = np.geomspace(0.02, 30, 200) * u.day
   L_bol = SEDClass.eval_bolometric(t, **params_grid)
   T = SEDClass.temperature(t, **params_grid)

   fig, (ax_L, ax_T) = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True)

   for row in range(n_samples):
       ax_L.plot(t.to_value(u.day), L_bol[row].to_value(u.erg / u.s), color="C0", lw=0.4, alpha=0.06)
       ax_T.plot(t.to_value(u.day), T[row].to_value(u.K), color="C3", lw=0.4, alpha=0.06)

   ax_L.set_xscale("log")
   ax_L.set_yscale("log")
   ax_L.set_ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
   ax_L.set_title("Kilonova: simulated bolometric light curves (n=1000)")

   ax_T.set_yscale("log")
   ax_T.set_xlabel("Time since merger [days]")
   ax_T.set_ylabel("Photospheric temperature [K]")

   fig.tight_layout()

References
-----------

.. footbibliography::
