.. _transients_supernovae:

Type IIP Supernovae
=====================

Type IIP core-collapse supernovae (CCSNe) are the explosions of hydrogen-rich massive stars whose
extended envelopes produce a characteristic weeks-to-months-long luminosity "plateau" as a
recombination front recedes through the ejecta. Two populations are implemented here: ordinary
Type IIP SNe, and a subset of Type IIP SNe that show a distinct early-time (first few days)
luminosity excess, attributed to shock breakout through and/or collisional heating of confined
circumstellar material (CSM) shed by the progenitor shortly before explosion. SN 2023ixf :footcite:p:`hsu2025` and SN 2024ggi :footcite:p:`chen2024` are the best-observed
examples of this early-interacting subclass, and both show a fast early rise to a hot
(:math:`\sim2.5\times10^4` K) peak within the first few days, followed by cooling toward the
ordinary IIP plateau temperature. UVEX's UV sensitivity makes it well suited to
catching precisely this brief, hot early phase, which fades quickly out of optical bands. Both
populations are implemented by :class:`~uvex_transients.transients.supernovae.TypeIIPSNe` and
:class:`~uvex_transients.transients.supernovae.TypeIIPExcessSNe`, pairing
:class:`~uvex_transients.models.supernovae.IIp_excess.TypeIIPSED` and
:class:`~uvex_transients.models.supernovae.IIp_excess.TypeIIPExcessSED` respectively with the rate/duration
metadata described below.

Transient Rates
----------------

Both populations are drawn as fixed fractions of a single, shared total core-collapse rate,
:math:`R_\mathrm{CC}(z) = k h^2 \psi_\mathrm{UV}(z)`, where :math:`\psi_\mathrm{UV}(z)` is the
UV-derived cosmic star-formation-rate density of :footcite:t:`madau2014` and the normalization
:math:`k` is fit directly to CCSNe counts by :footcite:t:`strolger2015` using the CANDELS and CLASH
Hubble Space Telescope supernova surveys out to :math:`z\approx2.5`. Tying the rate to a
directly-fit :math:`R_\mathrm{CC}(z)` (rather than, e.g., a fixed local value) lets the CCSNe rate
evolve self-consistently with cosmic star formation across the redshift range UVEX is sensitive to.

Of this total CCSNe rate, ordinary Type IIP SNe make up 40%, the local Type IIP fraction of the
core-collapse population reported from the volume-limited Lick Observatory Supernova Search sample
:footcite:p:`li2011`. Of *those* Type IIP SNe, roughly 30% show the SN 2023ixf/SN
2024ggi-like early interaction signature, so the early-interacting subclass is assigned
:math:`0.40\times0.30=12\%` of the total CCSNe rate. The
ordinary IIP population uses a redshift limit of :math:`z=1`; the early-interacting population, on
account of its higher peak temperatures and correspondingly bluer, more UV-detectable emission, is
given a more generous limit of :math:`z=2`. Both use a 200-day duration window, generous enough to
cover the plateau and the transition to the (unmodeled) nebular phase.

.. list-table::
   :header-rows: 1
   :widths: 30 20 15 15 20

   * - Population
     - Fraction of :math:`R_\mathrm{CC}(z)`
     - Redshift Limit
     - Duration
     - Source
   * - Type IIP (:class:`~uvex_transients.transients.supernovae.TypeIIPSNe`)
     - 40%
     - :math:`z = 1`
     - 200 days
     - :footcite:t:`strolger2015`, :footcite:t:`madau2014`, :footcite:t:`li2011`
   * - Type IIP + early excess (:class:`~uvex_transients.transients.supernovae.TypeIIPExcessSNe`)
     - 12%
     - :math:`z = 2`
     - 200 days
     - :footcite:t:`strolger2015`, :footcite:t:`madau2014`, :footcite:t:`li2011`

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
the ordinary Type IIP model, then for the Type IIP + early-excess variant.

.. plot::
   :include-source: false

   import numpy as np
   import matplotlib.pyplot as plt
   from astropy import units as u

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

   ax_L.set_xscale("log")
   ax_L.set_yscale("log")
   ax_L.set_ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
   ax_L.set_title("Type IIP SNe: simulated bolometric light curves (n=1000)")

   ax_T.set_yscale("log")
   ax_T.set_xlabel("Time since explosion [days]")
   ax_T.set_ylabel("Photospheric temperature [K]")

   fig.tight_layout()

.. plot::
   :include-source: false

   import numpy as np
   import matplotlib.pyplot as plt
   from astropy import units as u

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

   ax_L.set_xscale("log")
   ax_L.set_yscale("log")
   ax_L.set_ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
   ax_L.set_title("Type IIP + early excess SNe: simulated bolometric light curves (n=1000)")

   ax_T.set_yscale("log")
   ax_T.set_xlabel("Time since explosion [days]")
   ax_T.set_ylabel("Photospheric temperature [K]")

   fig.tight_layout()

References
-----------

.. footbibliography::
