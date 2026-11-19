"""Model of the SED from type Ic-BL (broad-lined, stripped-envelope) supernovae."""

from typing import ClassVar

from astropy import units as u

from uvex_transients.models.core.parameters import Parameter
from uvex_transients.models.core.priors import TruncatedNormalPrior
from uvex_transients.models.supernovae.Ibc import TypeIcSED

__all__ = ["TypeIcBLSED"]


class TypeIcBLSED(TypeIcSED):
    r"""
    Type Ic-BL (broad-lined) supernova SED, with priors from the ZTF volume-limited stripped-envelope SN sample.

    This is :class:`~uvex_transients.models.supernovae.Ibc.TypeIcSED` with its own ``M_Ni``, ``M_ej`` and ``v_ej``,
    tuned to the SNe Ic-BL of the ZTF CLU + CATS150 sample [citation pending] in the same way (each truncated normal
    is tuned so that its 16th, 50th and 84th percentiles, after truncation, reproduce the sample's):

    * ``M_Ni``: median :math:`0.19\,M_\odot`, 16th to 84th percentile 0.093 to 0.29 :math:`M_\odot`.
    * ``M_ej``: the observed Ic-BL median of about :math:`3.2\,M_\odot` (16th to 84th percentile 1.3 to 5.4)
      scaled by 0.90, the sample-wide shift in median ejecta mass under survey-efficiency weighting (see
      :class:`~uvex_transients.models.supernovae.Ibc.TypeIbSED`), giving 1.17, 2.9 and 4.86 :math:`M_\odot`.
    * ``v_ej``: :math:`16.8\times10^3` km/s with a width of :math:`3.5\times10^3` km/s. This is the CLU velocity
      proxy (see :class:`~uvex_transients.models.supernovae.Ibc.TypeIbSED`), not the photospheric velocities of
      :footcite:t:`srinivasaragavan2024`, which this model used before and which have a mean of
      :math:`20.1\times10^3` km/s.

    Every other prior is that of Type Ic (``kappa`` is :math:`0.07\,\mathrm{cm^2\,g^{-1}}` throughout), including
    the UV line blanketing set against Swift/UVOT and the host dust: see that class and
    :class:`~uvex_transients.models.supernovae.Ibc.TypeIbSED` for how they were chosen, and for what the data do and
    do not constrain. A change to those Type Ic priors is a change to this model too. The ``M_Ni`` of this model is
    not the log-normal of Types Ib and Ic, which was set against the ASAS-SN luminosity function of those types.

    .. rubric:: Parameters

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``M_Ni``
         - :math:`M_\mathrm{Ni}`
         - Nickel-56 mass synthesized in the explosion. The Ic-BL sample's.
       * - ``M_ej``
         - :math:`M_\mathrm{ej}`
         - Ejecta mass. The Ic-BL sample's.
       * - ``v_ej``
         - :math:`v_\mathrm{ej}`
         - Ejecta velocity, taken to be constant and equal to the photospheric velocity. The Ic-BL sample's.
       * - ``kappa``
         - :math:`\kappa`
         - Grey optical opacity. As Type Ic.
       * - ``kappa_gamma``
         - :math:`\kappa_\gamma`
         - Opacity to high-energy photons. As Type Ic.
       * - ``T_floor``
         - :math:`T_\mathrm{floor}`
         - Minimum photospheric temperature. As Type Ic.
       * - ``break_frequency``
         - :math:`\nu_\mathrm{break}`
         - Frequency blueward of which line blanketing suppresses the flux. As Type Ic.
       * - ``beta``
         - :math:`\beta`
         - Power-law index of the suppression above the break. As Type Ic.
       * - ``A_V``
         - :math:`A_V`
         - V-band attenuation by dust in the host galaxy. As Type Ic.
       * - ``R_V``
         - :math:`R_V`
         - Total-to-selective extinction ratio of the host dust. As Type Ic.

    References
    ----------
    .. footbibliography::
    """

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        **TypeIcSED._DEFAULT_PARAMETERS,
        "M_Ni": Parameter(
            # Truncated quantiles: 0.093 [16%], 0.189 [50%], 0.291 [84%] Msun (target 0.093, 0.19, 0.29).
            prior=TruncatedNormalPrior(mean=0.183, sigma=0.106, lower=0.005, upper=1.5),
            scale=1 * u.Msun,
            description="Nickel-56 mass synthesized in the explosion.",
            latex=r"M_\mathrm{Ni}",
        ),
        "M_ej": Parameter(
            # Truncated quantiles: 1.18 [16%], 2.87 [50%], 4.89 [84%] Msun (target 1.17, 2.9, 4.86).
            prior=TruncatedNormalPrior(mean=2.47, sigma=2.22, lower=0.1, upper=20.0),
            scale=1 * u.Msun,
            description="Ejecta mass.",
            latex=r"M_\mathrm{ej}",
        ),
        "v_ej": Parameter(
            # Truncated quantiles: 13.3 [16%], 16.8 [50%], 20.3 [84%] (target 13.3, 16.8, 20.3).
            prior=TruncatedNormalPrior(mean=16.8, sigma=3.5, lower=5.0, upper=30.0),
            scale=1e3 * u.km / u.s,
            description="Constant ejecta velocity, identified with the photospheric velocity.",
            latex=r"v_\mathrm{ej}",
        ),
    }
