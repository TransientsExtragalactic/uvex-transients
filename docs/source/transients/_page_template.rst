.. This is a TEMPLATE, not a built documentation page (see the `exclude_patterns`
   entry in conf.py). Copy it to a new file in this directory, fill in every
   ``<...>`` placeholder, and add the new file to the toctree in ``index.rst``.
   See kilonovae.rst / tdes.rst / lfbots.rst / supernovae.rst for filled-in
   examples of every section below.

.. _transients_<slug>:

<Transient Class Name>
=======================

<One short paragraph (or two): what the transient physically is, why it is astrophysically
interesting, and the observational picture that motivates the model choice below -- keep this
tight, the details belong in the sections that follow, not the overview. Cite the
defining/prototype observations with :footcite:t:`key` (for an in-sentence citation, e.g.
":footcite:t:`waxman2018` show...") or :footcite:p:`key` (for a parenthetical citation, e.g.
"...(the GW170817 kilonova; :footcite:p:`cowperthwaite2017`)"). Close with a sentence naming the
concrete :mod:`uvex_transients` classes involved, e.g. "This population is implemented by
:class:`~uvex_transients.transients.<module>.<TransientClass>`, pairing
:class:`~uvex_transients.models.<subpackage>.<SEDClass>` with the rate/duration metadata described
below.">

Quick Facts
------------

<A single ``list-table`` capturing the rate, redshift limit, and duration at a glance -- one row
each, with the literature discussion and reasoning moved into the ``Notes`` column rather than
narrative paragraphs above the table. If a transient class covers more than one population (e.g.
several subtypes sharing a rate model), still use one row per quantity but note the per-population
split inline in ``Value``/``Notes`` (see supernovae.rst for an example). Use ``--`` in ``Source``
for rows without a direct literature citation (e.g. a redshift limit chosen by design rather than
drawn from a paper).>

.. list-table::
   :header-rows: 1
   :widths: 15 25 15 45

   * - Quantity
     - Value
     - Source
     - Notes
   * - Rate
     - <value, e.g. "$10\ \mathrm{Gpc^{-3}\,yr^{-1}}$ (constant in $z$)">
     - :footcite:t:`<key>`
     - <Which literature estimate(s) were considered, which was adopted and why, and whether/how
       it evolves with redshift.>
   * - Redshift limit
     - <z_max>
     - --
     - <The argument for this limit -- e.g. a luminosity/detectability argument, or a generous
       bound relative to the population's own timescales.>
   * - Duration
     - <duration>
     - --
     - <Why this window safely bounds the transient's total observable duration.>

SED Model
----------

<Describe the physical/phenomenological model choice: what functional form the
bolometric light curve takes, whether/how the photospheric temperature
evolves, and why this parameterization was chosen for this transient class.
Reference :class:`~uvex_transients.models.<subpackage>.<SEDClass>` and cite the
paper(s) the functional form and/or priors are drawn from.>

The model parameters and their default priors are as follows.

.. list-table::
   :header-rows: 1
   :widths: 16 12 40 32

   * - Parameter
     - Symbol
     - Prior
     - Notes / Source
   * - ``<param_name>``
     - :math:`<latex>`
     - <Distribution>(<parameters>)
     - :footcite:t:`<key>`

Simulated Light Curves
~~~~~~~~~~~~~~~~~~~~~~~

The plot below draws ~1000 random parameter realizations from the priors
above and shows the resulting bolometric light curves<, and photospheric
temperatures (if the model is thermal)>. <If real comparison data for the
calibrating event(s) exists in the packaged light-curve archive
(``test_data/transients/lightcurves.h5``, via
:class:`~uvex_transients.utils.lightcurve_archive.LightcurveArchive`), overlay it as a
scatter series on top of the simulated realizations (see kilonovae.rst); omit
this if no such data is available.>

.. plot::
   :include-source: false

   <matplotlib/numpy code sampling `SEDClass().sample_parameters(size=1000, ...)`
   and evaluating `SEDClass.eval_bolometric` / `SEDClass.temperature` over a
   time grid spanning this class's `DEFAULT_DURATION`>


----

Observability Summary
----------------------

<Two ``.. plot::`` blocks, using the corresponding `~uvex_transients.transients.<module>.<TransientClass>`
(not just the SED class) to draw a Monte Carlo realization of the population via
`sample_event_redshift`/`sample_parameters`, and `m4opt.missions.uvex.detector.bandpasses` for the
UVEX FUV/NUV bandpasses:

1. Peak apparent AB magnitude vs. redshift, per band, with a scatter of simulated events, a
   Gaussian KDE contour, and the UVEX 1-Dwell limit (:math:`m<24.5`) overplotted. Note in the
   intro text what redshift limit this justifies confidence in.

   To find each event's peak apparent magnitude: if the SED has a single parameter (or fixed
   combination, e.g. LFBOTs' ``t_peak`` or TDEs' ``5 * sigma_rise``) that is *exactly* the
   bolometric peak by construction, evaluate at that one observed-frame time
   (``t_obs_peak = t_peak_rest * (1 + z)``) as in kilonovae.rst/lfbots.rst/tdes.rst. If no such
   parameter exists -- e.g. a multi-component model where a named "reference" time is not the
   true peak, or a plateau/excess light curve -- numerically search a time grid instead (evaluate
   `mag_bandpass` over a broadcasted ``(n_events, n_times)`` grid and take ``np.nanmin`` over the
   time axis), as in supernovae.rst. Don't guess which applies: check the `Lightcurve` class's own
   docstring for whether its peak time is exact.

2. An all-sky "peak-visible rate" bar chart per band, assuming any event above :math:`m<24.5` is
   detectable and the population is isotropic/homogeneous in comoving volume out to its redshift
   limit: ``transient.all_sky_rate`` gives the all-sky rate directly, and the fraction of
   simulated events with peak magnitude below the limit scales it down to a peak-visible rate. Draw
   this with `~uvex_transients.utils.plotting.plot_rate_bars` rather than a bare ``ax.bar`` call, so
   the bars carry the same two uncertainty layers as a detection funnel: count
   ``visible_counts[band_name] = int(np.count_nonzero(visible))`` (not ``np.mean``) per band, and
   pass ``n_samples``, ``all_sky_rate``, and ``rate_ci=transient.RATE_CI`` through so
   `plot_rate_bars` can draw MC (statistical, Clopper-Pearson on ``visible_counts``) and rate
   (systematic, from `~uvex_transients.transients.base.ExtragalacticTransient.RATE_CI`) bands; color
   each bar with `~uvex_transients.utils.plotting.get_band_color` and finish with
   `~uvex_transients.utils.plotting.add_funnel_legend` so the two layers are labeled. See
   kilonovae.rst/tdes.rst/lfbots.rst/slsne.rst for the single-population form.

If a transient class covers more than one population (e.g. several subtypes), repeat each plot
once per population, including its own `plot_rate_bars` call with that subtype's own ``RATE_CI``
(see type_i.rst/type_ii.rst) rather than duplicating the whole section per subtype.>

References
-----------

.. footbibliography::
