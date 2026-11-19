"""Models of SEDs from type IIb supernovae and associated scenarios."""

from typing import ClassVar

import numpy as np
from astropy import units as u

from uvex_transients.models.core.parameters import Parameter
from uvex_transients.models.core.priors import LogNormalPrior, TruncatedNormalPrior
from uvex_transients.models.shock_cooling.morag import MoragShockCoolingArnettSED, MoragShockCoolingSED
from uvex_transients.models.supernovae.Ibc import _TypeIbIntrinsicSED

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
    :class:`~uvex_transients.models.shock_cooling.morag.MoragShockCoolingSED`, except ``radius`` and
    ``envelope_mass``, whose widths (:math:`\sigma_{\ln}=1.1` and 1.3, against 0.85 and 1.0) are increased so that the
    spread of the early UV peaks covers that of the Swift/UVOT sample. The nickel priors are those of
    :class:`~uvex_transients.models.supernovae.Ibc.TypeIbSED` (without its host dust: this model has none), except
    ``M_Ni``, a log-normal truncated to 0.005 to 1.5 :math:`M_\odot` (median :math:`0.02\,M_\odot`,
    :math:`\sigma_{\ln}=1.1`) set against the ASAS-SN luminosity function of SNe IIb. The line blanketing
    (``break_frequency``, ``beta``) is that of Type Ib, set against Swift/UVOT, and applies to the nickel component
    only. See
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

    _ARNETT_SED: ClassVar[type[_TypeIbIntrinsicSED]] = _TypeIbIntrinsicSED
    _ARNETT_PARAMETERS: ClassVar[tuple[str, ...]] = tuple(_TypeIbIntrinsicSED._DEFAULT_PARAMETERS)

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        **MoragShockCoolingSED._DEFAULT_PARAMETERS,
        **_TypeIbIntrinsicSED._DEFAULT_PARAMETERS,
        # Wider than the Morag+24 priors (sigma_ln 0.85 and 1.0), so that the spread of the early UV peaks covers the
        # Swift/UVOT Type IIb sample, whose peaks span about 3 dex. The mean of each is unchanged.
        "radius": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=1.1),
            scale=6e12 * u.cm,
            description="Progenitor stellar radius.",
            latex=r"R",
        ),
        "envelope_mass": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=1.3),
            scale=0.05 * u.Msun,
            description="Hydrogen-rich envelope mass.",
            latex=r"M_E",
        ),
        "M_Ni": Parameter(
            # A normal in ln(M_Ni / Msun), truncated. Quantiles: 0.009 [16%], 0.023 [50%], 0.065 [84%], 0.18 [97.5%].
            prior=TruncatedNormalPrior(mean=np.log(0.02), sigma=1.1, lower=np.log(0.005), upper=np.log(1.5)),
            scale=1 * u.Msun,
            transform="log",
            description="Nickel-56 mass synthesized in the explosion (log-normal, set against the ASAS-SN LF).",
            latex=r"M_\mathrm{Ni}",
        ),
    }
