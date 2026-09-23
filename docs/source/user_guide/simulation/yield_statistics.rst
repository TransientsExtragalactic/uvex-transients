.. _yield-statistics:

Yield Statistics
================

A simulated event catalog is one realization of a transient population and
its observation by a survey. Its detected event count is only a *realized* yield and not
a formal estimate of the expected yield.

This note defines the **yield estimate**: the analytically expected number
of intrinsic events multiplied by the detection efficiency measured from
the catalog. It establishes confidence bounds for this estimate, including
uncertainty in the astrophysical rate normalization. All intervals are
computed from distribution quantiles or confidence-set transformations;
no symmetric-error approximation is used.

Conventions and notation
------------------------

We consider one transient population, one fixed survey schedule, and one
specified detection criterion. The cosmology, redshift evolution, and
distribution of transient properties are held fixed. Only the rate
normalization is uncertain in the population model considered here.

.. list-table:: Notation
   :header-rows: 1
   :widths: 18 29 53

   * - Symbol
     - Name
     - Definition
   * - :math:`R(z;A)`
     - Volumetric event rate
     - Events per comoving volume per source-frame time.
   * - :math:`A`
     - Rate normalization
     - Unknown physical normalization of the volumetric rate.
   * - :math:`R_0`
     - Fiducial rate normalization
     - Adopted central estimate of :math:`A`, used to generate the catalog.
   * - :math:`f(z)`
     - Rate evolution
     - Fixed, nonnegative, dimensionless redshift dependence.
   * - :math:`\mathcal R(A)`
     - All-sky event rate
     - Expected events per observer-frame time in the redshift domain.
   * - :math:`\mathcal V`
     - Rate-weighted comoving volume
     - Full-sky volume integral including evolution and time dilation.
   * - :math:`\mathcal E`
     - Population exposure
     - :math:`T\mathcal V`, with units of volume times time.
   * - :math:`\mu(A)`
     - Expected intrinsic count
     - :math:`A\mathcal E` in the population sampling domain.
   * - :math:`\epsilon`
     - Detection efficiency
     - Probability that an event sampled from that domain is detected.
   * - :math:`\lambda(A,\epsilon)`
     - Expected detected yield
     - :math:`A\mathcal E\epsilon`.
   * - :math:`n`
     - Simulated event count
     - Number of intrinsic events in the realized catalog.
   * - :math:`k`
     - Detected event count
     - Number of unique catalog events satisfying the detection criterion.
   * - :math:`\widehat\epsilon`
     - Estimated detection efficiency
     - :math:`k/n`, when :math:`n>0`.
   * - :math:`\widehat\lambda`
     - Yield estimate
     - :math:`R_0\mathcal E\widehat\epsilon`.

Lowercase :math:`n` and :math:`k` denote realized integer counts. Greek
:math:`\mu` and :math:`\lambda` denote expected counts, which need not be
integers. Hats denote estimators, and subscripts :math:`\mathrm L` and
:math:`\mathrm U` denote lower and upper bounds. A subscript :math:`0`
denotes evaluation at the fiducial normalization :math:`R_0`.

Every confidence interval carries a stated confidence level
:math:`C=1-\alpha`. A frequentist confidence level describes the coverage
of the interval-construction procedure under repeated sampling; it is
not a probability distribution assigned to a fixed unknown parameter.

The population rate
-------------------

We write the **intrinsic volumetric rate** (events / comoving volume / rest-frame time) as

.. math::

   R(z;A)
   \equiv \frac{dN}{dV_c\,dt_{\rm src}}
   = A f(z).

Here, :math:`A` is the **physical rate normalization**, with units
:math:`\mathrm{Mpc}^{-3}\,\mathrm{yr}^{-1}`, and :math:`f(z)` is the redshift dependence of the rate.

.. tip::

    The convention :math:`f(0)=1` makes :math:`A` the local volumetric rate, but another fixed normalization
    is equally valid.

In general, these intrinsic rates are taken from the literature and are reported with a given uncertainty,

.. math::

   A\in[R_{\mathrm L},R_{\mathrm U}].

If a publication reports :math:`R_0 {}^{+\Delta R_+}_{-\Delta R_-}`, the
corresponding endpoints are

.. math::

   R_{\mathrm L}=R_0-\Delta R_-,
   \qquad
   R_{\mathrm U}=R_0+\Delta R_+.

.. admonition:: Convention

    Wherever relevant in the code-base, uncertainties in the intrinsic rate are taken to be the 90% confidence interval.

All-sky rate and intrinsic count
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cosmological time dilation gives

.. math::

   dt_{\rm obs}=(1+z)\,dt_{\rm src}.

Consequently, the differential observer-frame event rate is

.. math::

   \frac{d\dot N}{dz\,d\Omega}
   =\frac{A f(z)}{1+z}\frac{dV_c}{dz\,d\Omega}.

For an isotropic population in :math:`0\leq z\leq z_{\max}`, we define the **rate-weighted comoving volume** as

.. math::

   \mathcal V
   =4\pi\int_0^{z_{\max}}
   \frac{f(z)}{1+z}\frac{dV_c}{dz\,d\Omega}\,dz,
   \qquad
   \frac{dV_c}{dz\,d\Omega}
   =\frac{cD_M^2(z)}{H(z)},

where :math:`D_M` is the transverse comoving distance.

The **all-sky event rate** and its rate-only confidence bounds are

.. math::

   \mathcal R(A)=A\mathcal V,
   \qquad
   \mathcal R_0=R_0\mathcal V,
   \qquad
   [\mathcal R_{\mathrm L},\mathcal R_{\mathrm U}]
   =[R_{\mathrm L}\mathcal V,R_{\mathrm U}\mathcal V].

For an observer-frame sampling window of duration :math:`T`.
The expected all-sky yield and its rate-only bounds are then

.. math::

   \mu(A)=AT\mathcal V,
   \qquad \mu_0=R_0T\mathcal V,
   \qquad
   [\mu_{\mathrm L},\mu_{\mathrm U}]
   =[R_{\mathrm L}T\mathcal V,R_{\mathrm U}T\mathcal V].

The normalized event-redshift density is

.. math::

   p_z(z)=
   \frac{
     \displaystyle\frac{f(z)}{1+z}\frac{dV_c}{dz\,d\Omega}
   }{
     \displaystyle\int_0^{z_{\max}}
     \frac{f(z')}{1+z'}\frac{dV_c}{dz'\,d\Omega}\,dz'
   }.

Both the normalization :math:`A` and the full-sky solid angle cancel.
Changing only :math:`A` therefore changes the number of events but not
the distribution of their redshifts or other sampled properties.

Detection efficiency and catalog generation
-------------------------------------------

Let :math:`x` denote an event's redshift, sky position, reference time,
and intrinsic properties. Let :math:`p(x)` be their normalized joint
density in the sampling domain. For a fixed survey, define
:math:`s(x)\in[0,1]` as the probability that the event satisfies the
detection criterion. A deterministic selection is the special case
:math:`s(x)\in\{0,1\}`.

The population-averaged detection efficiency is

.. math::

   \epsilon=\int s(x)p(x)\,dx,
   \qquad 0\leq\epsilon\leq1,

and the expected detected yield is

.. math::

   \lambda(A,\epsilon)=\mu(A)\epsilon
   =A T\mathcal V\epsilon.

This efficiency includes the footprint, cadence, observing depth,
extinction treatment, light-curve model, and detection criterion.
Do not apply an additional sky-coverage factor if it is already included
in :math:`\epsilon`.

At the fiducial normalization, the simulation draws

.. math::

   n\sim\operatorname{Poisson}(\mu_0).

Conditional on :math:`n`, independently sampled events and independent
event-level detection outcomes give

.. math::

   k\mid n\sim\operatorname{Binomial}(n,\epsilon),

   P(k\mid n,\epsilon)
   =\binom nk\epsilon^k(1-\epsilon)^{n-k}.

Individual events may have different detectabilities. The binomial model
holds because their properties are independent draws from the same
population and those properties are averaged over in :math:`\epsilon`.
Conditional on a fixed list of unequal detection probabilities, the
count instead follows a Poisson-binomial distribution.

Marginalizing over the Poisson-distributed intrinsic count gives

.. math::

   k\sim\operatorname{Poisson}(\lambda_0),
   \qquad \lambda_0=\mu_0\epsilon.

This is Poisson thinning. It assumes that detecting one event does not
alter the selection of another. Shared random survey conditions,
population-dependent follow-up competition, and correlations between
events require additional modeling. The present bounds are conditional
on the specified survey and population shape.

Estimating the expected yield
-----------------------------

For :math:`n>0`, the conditional binomial maximum-likelihood estimate is

.. math::

   \widehat\epsilon=\frac{k}{n}.

The yield estimate is

.. math::

   \boxed{
   \widehat\lambda
   =\mu_0\widehat\epsilon
   =R_0\mathcal E\frac{k}{n}.
   }

At fixed :math:`R_0`, this estimator is conditionally unbiased:

.. math::

   \mathbb E[\widehat\epsilon\mid n]=\epsilon,
   \qquad
   \mathbb E[\widehat\lambda\mid n]=\lambda_0,
   \qquad n>0.

The raw detected count :math:`k` is also an unbiased estimator of
:math:`\lambda_0` under Poisson catalog generation. The efficiency-based
estimator uses the known analytic normalization :math:`\mu_0` and the
observed denominator :math:`n`, rather than retaining fluctuations in
the total simulated population count.

For example, if :math:`\mu_0=1000` and :math:`n=k=917`, then
:math:`\widehat\lambda=1000`. This does not establish that
:math:`\epsilon=1` exactly: a finite catalog in which all events are
detected still permits efficiencies below unity.

The same estimator works when :math:`n` is fixed deliberately, or when
more events are simulated to improve precision. In those cases,
:math:`\mu_0` remains the target population's expected intrinsic count;
it is not replaced by the larger simulation sample size. For independent
catalogs sampling the same :math:`p(x)` and selection, pool their counts:

.. math::

   n=\sum_j n_j,
   \qquad k=\sum_j k_j.

Confidence bounds from the simulated catalog
--------------------------------------------

For a requested simulation-only confidence level
:math:`C_\epsilon=1-\alpha_\epsilon`, use the central Clopper--Pearson
binomial interval. Write :math:`Q_{\rm B}(q;a,b)` for the quantile at
probability :math:`q` of a Beta distribution with shape parameters
:math:`a,b`. Then

.. math::

   \epsilon_{\mathrm L}=
   \begin{cases}
     0, & k=0,\\
     Q_{\rm B}(\alpha_\epsilon/2;k,n-k+1), & k>0,
   \end{cases}

   \epsilon_{\mathrm U}=
   \begin{cases}
     1, & k=n,\\
     Q_{\rm B}(1-\alpha_\epsilon/2;k+1,n-k), & k<n.
   \end{cases}

These bounds invert binomial tail tests. Their coverage is at least
:math:`1-\alpha_\epsilon` for every fixed :math:`\epsilon`; discreteness
generally makes the interval conservative. The Beta quantiles are a
computational identity, not an assumed Bayesian posterior. See the
`SciPy Clopper--Pearson documentation
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html>`_.

With the normalization held at :math:`R_0`, the corresponding
**simulation-only yield bounds** are

.. math::

   \boxed{
   [\lambda_{\mathrm L}^{\rm MC},\lambda_{\mathrm U}^{\rm MC}]
   =[\mu_0\epsilon_{\mathrm L},\mu_0\epsilon_{\mathrm U}].
   }

These bounds quantify uncertainty in the expected yield due to a finite
simulation sample. They are not prediction bounds for the count in a
future survey. Coverage also holds for a Poisson-distributed :math:`n`,
because the binomial construction covers conditionally for each sample
size, including the empty-catalog convention below.

Boundary cases
~~~~~~~~~~~~~~

For no detections in a nonempty catalog, :math:`k=0<n`,

.. math::

   \widehat\epsilon=0,
   \qquad
   [\epsilon_{\mathrm L},\epsilon_{\mathrm U}]
   =\left[0,1-(\alpha_\epsilon/2)^{1/n}\right].

The estimated yield is zero, but its upper confidence bound is positive.
Zero simulated detections do not establish a zero expected yield.

For detections of every simulated event, :math:`k=n>0`,

.. math::

   \widehat\epsilon=1,
   \qquad
   [\epsilon_{\mathrm L},\epsilon_{\mathrm U}]
   =\left[(\alpha_\epsilon/2)^{1/n},1\right].

For an empty catalog, :math:`n=k=0`, the detection efficiency is
unidentified. Return no point estimate and use

.. math::

   [\epsilon_{\mathrm L},\epsilon_{\mathrm U}]=[0,1],
   \qquad
   [\lambda_{\mathrm L}^{\rm MC},\lambda_{\mathrm U}^{\rm MC}]
   =[0,\mu_0].

Do not set :math:`\widehat\epsilon=0` when :math:`n=0`.

The bounds above are two-sided central intervals. A deliberately
one-sided upper bound at confidence :math:`1-\alpha_\epsilon` instead
uses

.. math::

   \epsilon_{\mathrm U}^{\rm one-sided}
   =Q_{\rm B}(1-\alpha_\epsilon;k+1,n-k),
   \qquad k<n,

with upper bound one when :math:`k=n`. In particular, for :math:`k=0<n`
it is :math:`1-\alpha_\epsilon^{1/n}`. The upper endpoint of a central
interval and a one-sided upper bound at the same stated confidence
level are different quantities.

Including rate uncertainty
--------------------------

Rate-only bounds
~~~~~~~~~~~~~~~~

Holding the estimated efficiency fixed gives the **rate-only yield
bounds**

.. math::

   [\lambda_{\mathrm L}^{R},\lambda_{\mathrm U}^{R}]
   =[\mathcal E\widehat\epsilon R_{\mathrm L},
     \mathcal E\widehat\epsilon R_{\mathrm U}].

For :math:`R_0>0`, equivalently,

.. math::

   [\lambda_{\mathrm L}^{R},\lambda_{\mathrm U}^{R}]
   =\widehat\lambda
   \left[\frac{R_{\mathrm L}}{R_0},
         \frac{R_{\mathrm U}}{R_0}\right].

This displays the rate contribution conditional on the fitted efficiency.
It does not account for uncertainty in that efficiency. In particular,
when :math:`k=0`, these rate-only plug-in bounds collapse to zero even
though the combined upper bound need not do so.

Combined bounds with guaranteed coverage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A joint frequentist interval can be constructed from separate confidence
sets without imposing a probability distribution on :math:`A`.
Choose a target combined confidence level :math:`C=1-\alpha` and allocate
the failure probability between the two inputs:

.. math::

   \alpha_\epsilon+\alpha_R\leq\alpha.

A simple default is :math:`\alpha_\epsilon=\alpha_R=\alpha/2`.
Compute the efficiency interval at :math:`1-\alpha_\epsilon` and obtain
the external rate interval at :math:`1-\alpha_R`. Both levels must be
available; do not reinterpret a published interval at a different level.

By the union bound, the rectangle

.. math::

   [\epsilon_{\mathrm L},\epsilon_{\mathrm U}]
   \times[R_{\mathrm L},R_{\mathrm U}]

covers the true pair :math:`(\epsilon,A)` with probability at least
:math:`1-\alpha_\epsilon-\alpha_R`. Since
:math:`\lambda=\mathcal E A\epsilon` is nondecreasing in both nonnegative
parameters, project the rectangle onto the expected yield:

.. math::

   \boxed{
   \lambda_{\mathrm L}
   =\mathcal E R_{\mathrm L}\epsilon_{\mathrm L},
   \qquad
   \lambda_{\mathrm U}
   =\mathcal E R_{\mathrm U}\epsilon_{\mathrm U}.
   }

Whenever the rectangle covers the true pair, this interval covers the
true expected yield. Thus its coverage is at least :math:`C`. This is
a conservative confidence construction, not an exact equal-tail
interval for the product. It requires no Gaussian approximation and
remains valid at :math:`k=0` or :math:`k=n`.

For example, a combined 95% interval can use a 97.5% efficiency interval
and a 97.5% rate interval. Multiplying the endpoints of two 95% intervals
does not by itself establish 95% combined coverage: the union-bound
guarantee is only 90%. Actual coverage can be higher.

The union-bound construction does not require independence. If the
two interval-coverage events are independent, the stronger lower bound
is

.. math::

   C_{\rm joint}\geq(1-\alpha_\epsilon)(1-\alpha_R).

Under that additional assumption, equal component levels
:math:`1-\alpha_\epsilon=1-\alpha_R=\sqrt C` suffice. Merely using
independent random seeds does not establish independence from every
external modeling uncertainty. The union-bound allocation is the
default here.

If a rate interval is available only at one confidence level, retain
that level and state the coverage guaranteed by the available component
intervals. Asymmetric endpoints alone cannot determine bounds at a new
confidence level. They also do not determine a unique likelihood or
sampling distribution. In that situation, report the simulation-only
and rate-only bounds separately, or report the combined interval with
its supported coverage guarantee.

For an empty catalog, the combined interval reduces to
:math:`[0,\mathcal E R_{\mathrm U}]`; there is still no efficiency-based
point estimate. All statements of coverage are conditional on the fixed
cosmology, evolution shape, and survey model.
