"""Models of SEDs from type IIb supernovae and associated scenarios."""

from typing import ClassVar

from astropy import units as u

from uvex_transients.models.core.parameters import Parameter
from uvex_transients.models.core.priors import ConstantPrior, TruncatedNormalPrior
from uvex_transients.models.shock_cooling.morag import MoragShockCoolingArnettSED, MoragShockCoolingSED
from uvex_transients.models.supernovae.Ibc import TypeIbSED

__all__ = ["TypeIIbSED"]


# ======================================== #
# Model: Shock Cooling + Nickel            #
# ======================================== #
class TypeIIbSED(MoragShockCoolingArnettSED):
    r"""
    Type IIb supernova SED: Morag+24 shock cooling plus a Type Ib-like nickel component.

    The sum of two emission components, with the spectral luminosities added,

    .. math::

        L_\nu(\nu, t) = L_\nu^\mathrm{shock}(\nu, t) + L_\nu^\mathrm{Ni}(\nu, t).

    The first is the full shock cooling SED of :footcite:t:`2024MNRAS.528.7137M`
    (:class:`~uvex_transients.models.shock_cooling.morag.MoragShockCoolingSED`, Eq. A7), evaluated at every epoch
    without the cut at the edges of that paper's regime of validity. It dominates the first days. The second is
    the line-blanketed, diffusion-limited nickel light curve of
    :class:`~uvex_transients.models.supernovae.Ibc.TypeIbSED` (:footcite:t:`1982ApJ...253..785A`), which produces
    the main peak at about 20 days. Together they give the Type IIb double structure: a luminous decline after
    breakout, a trough, then the rise to the nickel peak.

    The shock cooling priors are those of
    :class:`~uvex_transients.models.shock_cooling.morag.MoragShockCoolingSED`, and the nickel priors, including
    the line-blanketing parameters ``break_frequency`` and ``beta``, are those of
    :class:`~uvex_transients.models.supernovae.Ibc.TypeIbSED`, except ``M_Ni``, which is lowered to a
    Type IIb-like truncated normal (mean :math:`0.1\,M_\odot`, :math:`\sigma=0.07\,M_\odot`, bounds 0.01 to 0.4). See
    :class:`~uvex_transients.models.shock_cooling.morag.MoragShockCoolingArnettSED` for how the two components are
    combined.

    .. rubric:: Parameters

    The five parameters of :class:`~uvex_transients.models.shock_cooling.morag.MoragShockCoolingSED` (``v_star``,
    ``radius``, ``opacity``, ``envelope_mass``, ``core_mass``) and the eight of
    :class:`~uvex_transients.models.supernovae.Ibc.TypeIbSED` (``M_Ni``, ``M_ej``, ``v_ej``, ``kappa``,
    ``kappa_gamma``, ``T_floor``, ``break_frequency``, ``beta``).

    Notes
    -----
    The shock cooling formulas are not valid before :math:`t_\mathrm{min}`, where the luminosity grows without
    bound toward :math:`t \to 0`. The two components are independent, so nothing ties the envelope mass of the
    shock cooling to the ejecta mass of the nickel component.

    References
    ----------
    .. footbibliography::
    """

    _ARNETT_SED: ClassVar[type[TypeIbSED]] = TypeIbSED
    _ARNETT_PARAMETERS: ClassVar[tuple[str, ...]] = tuple(TypeIbSED._DEFAULT_PARAMETERS)

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        **MoragShockCoolingSED._DEFAULT_PARAMETERS,
        **TypeIbSED._DEFAULT_PARAMETERS,
        "M_Ni": Parameter(
            prior=TruncatedNormalPrior(mean=0.1, sigma=0.07, lower=0.01, upper=0.4),
            scale=1 * u.Msun,
            description="Nickel-56 mass synthesized in the explosion.",
            latex=r"M_\mathrm{Ni}",
        ),
        "beta": Parameter(
            prior=ConstantPrior(value=4.0),
            scale=1.0,
            description="Power-law index of the suppression above the break.",
            latex=r"\beta",
        ),
    }
