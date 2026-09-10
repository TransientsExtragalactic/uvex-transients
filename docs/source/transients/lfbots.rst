.. _transients_lfbots:

Luminous Fast Blue Optical Transients
========================================

Luminous fast blue optical transients (LFBOTs), typified by the prototype AT2018cow, are a rare
class of extragalactic transient distinguished by their unusually fast rise and decline
(:math:`\lesssim10` day timescales), persistently blue colors, and, in several cases, luminous
non-thermal X-ray and radio emission indicating a central engine (an accreting black hole or a
young magnetar) rather than radioactive decay powers the explosion. Their progenitor is still
debated -- proposed channels include failed supernovae, a white dwarf or star tidally disrupted by
an intermediate-mass black hole, and the merger of a compact object with a massive companion --
but their observed optical/UV emission is well described phenomenologically as a rapidly-evolving,
cooling blackbody photosphere :footcite:p:`holu2026`. This population is implemented by
:class:`~uvex_transients.transients.LFBOTs.LuminousFastBlueOpticalTransient`, pairing
:class:`~uvex_transients.models.lfbots.lfbots.LFBOTCoolingBlackbodySED` with the rate/duration metadata
described below.

Transient Rates
----------------

LFBOT volumetric rates reported in the literature span more than two orders of magnitude,
reflecting both small-number statistics (only a handful of confirmed events) and differing
selection criteria across surveys: :footcite:t:`coppejans2020` report rates of
:math:`<300\ \mathrm{Gpc^{-3}\,yr^{-1}}` from PTF and :math:`700`-:math:`1400\ \mathrm{Gpc^{-3}\,yr^{-1}}`
from PS1-MDS; the theoretical delayed-dynamical-instability model of :footcite:t:`klencki2025`
predicts :math:`15`-:math:`300\ \mathrm{Gpc^{-3}\,yr^{-1}}`, substantiated mainly by being
reasonably comparable to the observed rates; and :footcite:t:`perley2026` and
:footcite:t:`holu2026` report much lower rates of :math:`0.9`-:math:`12.5\ \mathrm{Gpc^{-3}\,yr^{-1}}`.
The rate adopted here, :math:`10\ \mathrm{Gpc^{-3}\,yr^{-1}}`, follows
:footcite:t:`perley2026` and :footcite:t:`holu2026` and is taken as constant with redshift, since
LFBOTs are too rare for their redshift evolution to yet be meaningfully constrained. A redshift
limit of :math:`z=4` and a 100-day duration window are used to safely bound the population relative
to the SED's own (much shorter) intrinsic rise/decline timescales.

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - Rate
     - Redshift Limit
     - Duration
     - Source
   * - :math:`10\ \mathrm{Gpc^{-3}\,yr^{-1}}` (constant in :math:`z`)
     - :math:`z = 4`
     - 100 days
     - :footcite:t:`perley2026,holu2026`

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
resulting bolometric light curves and photospheric temperatures.

.. plot::
   :include-source: false

   import numpy as np
   import matplotlib.pyplot as plt
   from astropy import units as u

   from uvex_transients.models.lfbots import LFBOTCoolingBlackbodySED as SEDClass

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

   ax_L.set_xscale("log")
   ax_L.set_yscale("log")
   ax_L.set_ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
   ax_L.set_title("LFBOTs: simulated bolometric light curves (n=1000)")

   ax_T.set_yscale("log")
   ax_T.set_xlabel("Time since explosion [days]")
   ax_T.set_ylabel("Photospheric temperature [K]")

   fig.tight_layout()

References
-----------

.. footbibliography::
