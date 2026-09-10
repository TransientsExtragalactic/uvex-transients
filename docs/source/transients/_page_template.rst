.. This is a TEMPLATE, not a built documentation page (see the `exclude_patterns`
   entry in conf.py). Copy it to a new file in this directory, fill in every
   ``<...>`` placeholder, and add the new file to the toctree in ``index.rst``.
   See kilonovae.rst / tdes.rst / lfbots.rst / supernovae.rst for filled-in
   examples of every section below.

.. _transients_<slug>:

<Transient Class Name>
=======================

<One-to-two paragraph overview: what the transient physically is, why it is
astrophysically interesting, and the observational picture that motivates the
model choice below. Cite the defining/prototype observations with
:footcite:t:`key` (for an in-sentence citation, e.g. ":footcite:t:`waxman2018`
show...") or :footcite:p:`key` (for a parenthetical citation, e.g. "...(the
GW170817 kilonova; :footcite:p:`cowperthwaite2017`)"). Name the concrete
:mod:`uvex_transients` classes involved, e.g. this population is implemented by
:class:`~uvex_transients.transients.<module>.<TransientClass>`, pairing
:class:`~uvex_transients.models.<subpackage>.<SEDClass>` with the rate/duration
metadata described below.>

Transient Rates
----------------

<Explain the volumetric rate :math:`R(z)` used for this population: the
literature value(s) it is drawn from, whether it is taken as constant in
redshift or evolving, and how the redshift limit ``DEFAULT_Z_LIM`` and duration
window ``DEFAULT_DURATION`` were chosen. Cite the rate paper(s). If multiple
rate estimates exist in the literature, explain which was adopted and why
(e.g. as a conservative choice).>

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Rate
     - Redshift Limit
     - Duration
     - Source
   * - <value>
     - <z_max>
     - <duration>
     - :footcite:t:`<key>`

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
temperatures (if the model is thermal)>.

.. plot::
   :include-source: false

   <matplotlib/numpy code sampling `SEDClass().sample_parameters(size=1000, ...)`
   and evaluating `SEDClass.eval_bolometric` / `SEDClass.temperature` over a
   time grid spanning this class's `DEFAULT_DURATION`>

References
-----------

.. footbibliography::
