.. _transients_llgrb:

Low-Luminosity GRBs
=====================

Low-luminosity gamma-ray bursts (llGRBs) are sub-energetic, long GRBs, typified by GRB 060218 (XRF 060218), which
exploded at :math:`z=0.0335` (:math:`d=145` Mpc) together with the broad-lined Type Ic SN 2006aj.
:footcite:t:`soderberg2006` find that such events are about 100 times less energetic than cosmological GRBs, with
:math:`\sim10^{48}` erg in mildly-relativistic ejecta, and about ten times more common. In UV and optical light an
llGRB is a two-part event: an early, hot, fast-fading thermal excess that the Swift/UVOT light curves of SN 2006aj
show peaking within the first day, which :footcite:t:`campana2006` interpret as the break-out of the shock
through the dense wind around the progenitor, followed by the radioactively powered supernova peaking about ten days
later. The model below has one component for each.

This population is implemented by
:class:`~uvex_transients.transients.llgrb.LowLuminosityGRB`, pairing
:class:`~uvex_transients.models.supernovae.llgrb.LLGRBSED` with the rate/duration metadata described below.

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
     - :math:`230\ \mathrm{Gpc^{-3}\,yr^{-1}}` (constant in :math:`z`)
     - :footcite:t:`soderberg2006`
     - The local rate of sub-energetic GRBs, with a 90% confidence range of :math:`40`-:math:`720\
       \mathrm{Gpc^{-3}\,yr^{-1}}` (+490/-190), adopted as
       :attr:`~uvex_transients.transients.llgrb.LowLuminosityGRB.RATE_CI` (see the dropdown below, and
       :ref:`user_guide_transients_rate_uncertainty` for the general mechanism). It is a local measurement that
       constrains no redshift evolution, so it is taken as constant in :math:`z`.
   * - Redshift limit
     - :math:`z = 2.41`
     - --
     - Redshift beyond which, with 95% confidence, at most 1% of events drawn from the SED's priors are
       brighter than 24.5 AB (the UVEX 1 Dwell limit) in either UVEX band, rounded up. It is set by the bright
       tail of the priors on the early excess, and the typical draw is detectable only to :math:`z\sim1`. See the
       observability summary below.
   * - Duration
     - 30 days
     - --
     - Generous: the early excess is bright for about a day, and fewer than 1% of events (weighted by
       volume) have any epoch brighter than 24.5 AB beyond 30 days.

.. dropdown:: Rate: what it is, and what it is not

   :footcite:t:`soderberg2006` estimate the rate from the two sub-energetic events caught by the
   gamma-ray missions that localize bursts, GRB 980425 (:math:`d=36` Mpc) and XRF 060218. For each of BeppoSAX
   (Wide Field Cameras), HETE-2 (Wide Field X-ray Monitor) and Swift (BAT) they compute how far away each
   event could have been detected (:math:`\sim60`-:math:`220` Mpc), and combine that volume with the instrument's
   sky coverage and operating time. The quoted uncertainty is dominated by the Poisson statistics of two
   detections.

   - **A true rate.** They argue that sub-energetic GRBs cannot be strongly beamed, since otherwise the rate
     would exceed the local rate of all Type Ibc supernovae, :math:`9^{+3}_{-5}\times10^{3}\
     \mathrm{Gpc^{-3}\,yr^{-1}}`. No beaming correction is applied here, and every event is sampled.
   - **Comparison.** It is about ten times the rate of typical bright GRBs, taking a mean inverse beaming
     factor :math:`\langle f_b^{-1}\rangle\sim100` for those. It is also comparable to the rate of broad-lined
     Ic supernovae, and the paper constrains events with strong early radio emission like these to
     :math:`\lesssim300\ \mathrm{Gpc^{-3}\,yr^{-1}}`, consistent with the llGRB rate.
   - **Not in the uncertainty range.** The range is statistical. It does not cover extrapolating a rate
     measured at :math:`d\lesssim200` Mpc to the redshift limit above, which is the largest systematic of the
     population, and the volume grows with the cube of the limit.

SED Model
----------

:class:`~uvex_transients.models.supernovae.llgrb.LLGRBSED` sums two blackbody components,

.. math::

    L_\nu(\nu, t) = L_\mathrm{SN}(t)\,S_\mathrm{BB}\!\left[\nu, T_\mathrm{SN}(t)\right]
    + L_\mathrm{bump}(t)\,S_\mathrm{BB}\!\left[\nu, T_\mathrm{bump}(t)\right],

where :math:`S_\mathrm{BB}` is a unit-integral blackbody shape. The supernova term :math:`L_\mathrm{SN}` is the
Arnett diffusion light curve of :class:`~uvex_transients.models.arnett.ArnettDecaySED`, powered by
:math:`^{56}\mathrm{Ni}` decay, with its floored photospheric temperature and no line blanketing. The early
excess is a Bazin pulse with a power-law photospheric temperature,

.. math::

    L_\mathrm{bump}(t) = A\,\frac{e^{-(t-t_0)/\tau_\mathrm{fall}}}{1 + e^{-(t-t_0)/\tau_\mathrm{rise}}},
    \qquad
    T_\mathrm{bump}(t) = T_1\left(\frac{t}{1\,\mathrm{d}}\right)^{-\alpha_T}.

For the default priors the Bazin maximum is before the burst, so the excess is already declining at
:math:`t=0`. The rise of the UV light curves over the first :math:`\sim0.5` d is the photosphere cooling, not a
rise in the bolometric luminosity.

The supernova priors are those of :class:`~uvex_transients.models.supernovae.IcBL.TypeIcBLSED`, the Ic-BL
population of :footcite:t:`srinivasaragavan2024` (see :ref:`transients_type_i`). They are not tuned to any llGRB.
The priors on the early excess are centered on a fit to the Swift/UVOT UVW2, UVM2 and UVW1 light curves of
SN 2006aj, the only GRB-associated supernova in the packaged Swift/UVOT archive. They are not a population
estimate, and their widths are judgment, not measured scatter.

.. list-table::
   :header-rows: 1
   :widths: 16 12 40 32

   * - Parameter
     - Symbol
     - Prior
     - Notes / Source
   * - ``M_Ni``
     - :math:`M_\mathrm{Ni}`
     - TruncatedNormal(mean=0.33, :math:`\sigma`\=0.24, 0.02-2) :math:`M_\odot`
     - Nickel-56 mass. Ic-BL population :footcite:p:`srinivasaragavan2024`.
   * - ``M_ej``
     - :math:`M_\mathrm{ej}`
     - TruncatedNormal(mean=2.54, :math:`\sigma`\=1.95, 0.1-10) :math:`M_\odot`
     - Ejecta mass. Ic-BL population.
   * - ``v_ej``
     - :math:`v_\mathrm{ej}`
     - TruncatedNormal(mean=20.1, :math:`\sigma`\=4.96, 5-45) :math:`10^3` km/s
     - Ejecta velocity, taken as the photospheric velocity. Ic-BL population.
   * - ``kappa``
     - :math:`\kappa`
     - Uniform(0.05, 0.15) :math:`\mathrm{cm^2\,g^{-1}}`
     - Grey optical opacity.
   * - ``kappa_gamma``
     - :math:`\kappa_\gamma`
     - Constant(1000) :math:`\mathrm{cm^2\,g^{-1}}`
     - High-energy opacity, fixed at a large value (full gamma-ray trapping).
   * - ``T_floor``
     - :math:`T_\mathrm{floor}`
     - TruncatedNormal(mean=6000, :math:`\sigma`\=1000, 3000-10000) K
     - Minimum photospheric temperature of the supernova.
   * - ``A_bump``
     - :math:`A`
     - Normal(:math:`\log_{10}(A/\mathrm{erg\,s^{-1}})`; mean=44.19, :math:`\sigma`\=0.3)
     - Bazin normalization of the early excess. Fit to SN 2006aj.
   * - ``t0_bump``
     - :math:`t_0`
     - Normal(:math:`\log_{10}(t_0/\mathrm{d})`; mean=-1.0, :math:`\sigma`\=0.25)
     - Bazin transition time. Unconstrained between 0.03 and 0.3 d by the fit, so the prior is wide.
   * - ``tau_rise``
     - :math:`\tau_\mathrm{rise}`
     - Normal(:math:`\log_{10}(\tau_\mathrm{rise}/\mathrm{d})`; mean=-0.34, :math:`\sigma`\=0.1)
     - Bazin rise timescale. Fit to SN 2006aj.
   * - ``tau_fall``
     - :math:`\tau_\mathrm{fall}`
     - Normal(:math:`\log_{10}(\tau_\mathrm{fall}/\mathrm{d})`; mean=-0.16, :math:`\sigma`\=0.1)
     - Bazin decline timescale. Fit to SN 2006aj.
   * - ``T_bump``
     - :math:`T_1`
     - Normal(:math:`\log_{10}(T_1/\mathrm{K})`; mean=4.21, :math:`\sigma`\=0.1)
     - Photospheric temperature of the excess at 1 d. Fit to SN 2006aj.
   * - ``alpha_T``
     - :math:`\alpha_T`
     - Normal(mean=0.48, :math:`\sigma`\=0.1)
     - Cooling power-law index of the excess. Fit to SN 2006aj.

.. dropdown:: Caveats of the calibration

   - **One event.** The early excess is calibrated to SN 2006aj alone. The fit reproduces its UVOT light curves to
     about 0.08 dex rms, with :math:`\kappa`, :math:`\kappa_\gamma` and :math:`T_\mathrm{floor}` held at their prior
     values. The supernova it implies, :math:`M_\mathrm{Ni}\simeq0.15`-:math:`0.2\,M_\odot` and
     :math:`M_\mathrm{ej}\simeq1.2\,M_\odot`, lies inside the Ic-BL priors, but only the late UV tail
     constrains it. :math:`v_\mathrm{ej}` is left free between :math:`1.7\times10^4` and :math:`4\times10^4` km/s.
   - **Luminosity and temperature are degenerate.** Before :math:`\sim0.2` d the excess is hot enough
     (:math:`T\gtrsim4\times10^4` K) that the UV bands lie on the Rayleigh-Jeans side of the blackbody, where
     :math:`L_\nu\propto L_\mathrm{bol}T^{-3}`. The UV fixes only that combination, so the bolometric luminosity
     of the excess, about :math:`8\times10^{43}` erg/s at the first UVOT epoch of 0.065 d, follows from the assumed
     power-law temperature. No bolometric data constrain it.
   - **No line blanketing in the supernova.** Its blackbody is not suppressed in the UV as the Type Ic-BL model's
     is, so its UV tail is set by the UVOT data of one event alone.
   - **Only the thermal emission.** The prompt gamma-ray and X-ray emission and the afterglow are not modeled, and
     there is no host or Milky Way extinction.

Simulated Light Curves
~~~~~~~~~~~~~~~~~~~~~~~

The plot below draws 500 random parameter realizations from the priors above and shows the bolometric light curve
and the photospheric temperature of each component. The early excess dominates for the first three days or so, and
the supernova takes over afterwards.

.. plot::
   :include-source: false

   import numpy as np
   import matplotlib.pyplot as plt
   from astropy import units as u

   from uvex_transients.models.arnett import ArnettDecaySED
   from uvex_transients.models.lightcurves.generic import BazinLightcurve
   from uvex_transients.models.supernovae import LLGRBSED

   rng = np.random.default_rng(20261020)
   n_samples = 500

   params = LLGRBSED().sample_parameters(size=n_samples, rng=rng)
   grid = {name: value[:, None] for name, value in params.items()}

   t = np.geomspace(0.03, 40, 200) * u.day
   L_total = LLGRBSED.eval_bolometric(t, **grid)
   L_sn = ArnettDecaySED.eval_bolometric(t, **{name: grid[name] for name in ArnettDecaySED._DEFAULT_PARAMETERS})
   L_bump = BazinLightcurve.eval(
       t, amplitude=grid["A_bump"], t0=grid["t0_bump"], rise=grid["tau_rise"], fall=grid["tau_fall"]
   )
   T_sn, T_bump = LLGRBSED.temperature(t, **grid)

   fig, (ax_L, ax_T) = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True)
   days = t.to_value(u.day)
   for row in range(n_samples):
       ax_L.plot(days, L_total[row].to_value(u.erg / u.s), color="k", lw=0.4, alpha=0.05)
       ax_L.plot(days, L_bump[row].to_value(u.erg / u.s), color="C3", lw=0.4, alpha=0.05)
       ax_L.plot(days, L_sn[row].to_value(u.erg / u.s), color="C0", lw=0.4, alpha=0.05)
       ax_T.plot(days, T_bump[row].to_value(u.K), color="C3", lw=0.4, alpha=0.05)
       ax_T.plot(days, T_sn[row].to_value(u.K), color="C0", lw=0.4, alpha=0.05)

   for color, label in [("k", "total"), ("C3", "early excess"), ("C0", "supernova")]:
       ax_L.plot([], [], color=color, label=label)
   ax_L.set_xscale("log")
   ax_L.set_yscale("log")
   ax_L.set_ylim(1e40, 1e46)
   ax_L.set_ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
   ax_L.set_title("llGRBs: simulated bolometric light curves (n=500)")
   ax_L.legend(loc="upper right", fontsize=8, frameon=False)

   ax_T.set_yscale("log")
   ax_T.set_xlabel("Time since explosion [days]")
   ax_T.set_ylabel("Photospheric temperature [K]")
   fig.tight_layout()

The second plot compares 100 draws with the Swift/UVOT UVW2, UVM2 and UVW1 light curves of SN 2006aj, from the
Swift Optical/Ultraviolet Supernova Archive (SOUSA), timed from the GRB trigger. Each draw is integrated through the
UVOT transmission curve from the SVO Filter Profile Service at the redshift of SN 2006aj, and both are shown as
:math:`\nu L_\nu` in the observed band with no K-correction. The early excess is calibrated to these data, so the
agreement is a check of the fit and not an independent test, and the spread of the draws is the width of the priors.

.. plot::
   :include-source: false

   import numpy as np
   import matplotlib.pyplot as plt
   from astropy import units as u

   from uvex_transients.models.supernovae import LLGRBSED
   from uvex_transients.utils.cosmology import get_cosmology
   from uvex_transients.utils.filter_utils import get_bandpass
   from uvex_transients.utils.swift_sne_archive import SwiftSNeArchive

   rng = np.random.default_rng(20261021)
   n_samples = 100

   archive = SwiftSNeArchive()
   redshift = float(archive.meta("SN2006aj")["redshift"])
   d_L = get_cosmology().luminosity_distance(redshift)

   params = LLGRBSED().sample_parameters(size=n_samples, rng=rng)
   grid = {name: value[:, None] for name, value in params.items()}
   t = np.geomspace(0.05, 40, 150) * u.day

   fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
   for ax, band, color in zip(axes, ["UVW2", "UVM2", "UVW1"], ["C3", "C1", "C0"]):
       nu_eff = (archive.meta("SN2006aj", band)["lambda_eff_AA"] * u.AA).to(u.Hz, equivalencies=u.spectral())
       mag = LLGRBSED.mag_bandpass(get_bandpass(f"Swift/UVOT.{band}"), t, redshift=redshift, **grid)
       nuLnu = (4 * np.pi * d_L**2 * nu_eff * mag.to(u.erg / u.s / u.cm**2 / u.Hz)).to_value(u.erg / u.s)
       ax.plot(t.to_value(u.day), nuLnu.T, color="0.4", lw=0.4, alpha=0.12)

       lc = archive.lightcurve("SN2006aj", band)
       ax.errorbar(lc["t"].value, lc["L"].value, lc["L_err"].value, fmt="o", ms=3, color=color, label="SN 2006aj")
       ax.set_xscale("log")
       ax.set_yscale("log")
       ax.set_xlabel("Time since GRB trigger [days]")
       ax.set_title(band)
       ax.legend(loc="upper right", fontsize=8, frameon=False)

   axes[0].set_ylim(1e40, 1e44)
   axes[0].set_ylabel(r"$\nu L_\nu$ [erg s$^{-1}$]")
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
magnitude limit. The redshift limit adopted above, :math:`z=2.41`, is its value at :math:`m=24.5`,
rounded up and derived on a finer grid than this plot's.

The limit bounds the fraction of the *population* that is detectable beyond it, not the fraction of
detected events, and it does not include Milky Way extinction or sky position. It is set by the bright tail of the
draws, which the broad priors on the amplitude and temperature of the early excess make long: the median draw
crosses the limit near :math:`z\sim1`, and only the brightest 1% reach the adopted redshift limit.

.. plot::
   :include-source: false

   import numpy as np
   import matplotlib.pyplot as plt

   from uvex_transients.missions import uvex_fast as uvex
   from uvex_transients.transients.llgrb import LowLuminosityGRB
   from uvex_transients.utils.plotting import plot_detection_horizon

   transient = LowLuminosityGRB()
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

The anticipated rate of llGRBs detectable by UVEX at these limits is as follows, assuming that any event whose
brightest epoch in the first 30 observer-frame days is above the :math:`m<24.5` limit is detectable, and that the
population is isotropic and homogeneous in comoving volume out to :math:`z=2.41`:

.. plot::
   :include-source: false

   import numpy as np
   import matplotlib.pyplot as plt
   from astropy import units as u

   from uvex_transients.missions import uvex_fast as uvex
   from uvex_transients.transients.llgrb import LowLuminosityGRB
   from uvex_transients.utils.plotting import add_funnel_legend, get_band_color, plot_rate_bars

   rng = np.random.default_rng(20261022)
   n_samples = 2000

   # Sample the llGRB population.
   llgrb = LowLuminosityGRB()
   redshift = llgrb.sample_event_redshift(n_samples, rng=rng)
   params = llgrb.sed.sample_parameters(size=n_samples, rng=rng)

   # The all-sky rate, with no survey footprint applied.
   all_sky_rate = llgrb.all_sky_rate

   # Observer-frame epochs at which each event's brightness is evaluated, to find its peak.
   t_obs = np.geomspace(0.03, 30, 50) * u.day

   detection_limits = {
       "FUV": 24.5 * u.ABmag,
       "NUV": 24.5 * u.ABmag,
   }

   # Compute the number of the n_samples draws visible in each band at their brightest epoch.
   visible_counts = {}

   for band_name, bandpass in uvex.detector.bandpasses.items():
       magnitudes = llgrb.sed.mag_bandpass(
           bandpass,
           t_obs[None, :],
           redshift=redshift[:, None],
           **{name: value[:, None] for name, value in params.items()},
       ).to_value(u.ABmag)

       visible = magnitudes.min(axis=1) < detection_limits[band_name].to_value(u.ABmag)
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
       rate_ci=llgrb.RATE_CI,
       color=[get_band_color(band) for band in band_names],
   )

   ax.set_yscale("log")
   ax.set_ylabel(r"All-sky rate [yr$^{-1}$]")
   ax.set_title("Peak-visible llGRB rate")
   add_funnel_legend(ax, loc="lower right")

   fig.tight_layout()
   plt.show()

These rates are upper bounds on what UVEX will see. They count every event that is above the limit at its
brightest epoch, with no survey cadence, footprint or exposure, so they say how many llGRBs are bright enough, not
how many are observed. They also extrapolate the local rate of :footcite:t:`soderberg2006` to :math:`z=2.41`: most
of the volume, and so most of the events, lie at :math:`z\gtrsim0.5`, far beyond the :math:`d<200` Mpc that
rate was measured in. The systematic band is the 90% range of that local rate only.


References
-----------

.. footbibliography::
