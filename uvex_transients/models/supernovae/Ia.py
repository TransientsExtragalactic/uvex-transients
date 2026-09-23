"""Model of the SED from type Ia (thermonuclear) supernovae."""

from typing import ClassVar

import numpy as np
from astropy import units as u

from uvex_transients.models.arnett import ArnettDecaySED
from uvex_transients.models.core.parameters import Parameter
from uvex_transients.models.core.priors import (
    ConstantPrior,
    MixturePrior,
    NormalPrior,
    TruncatedNormalPrior,
    UniformPrior,
)

__all__ = ["TypeIaSED"]


class TypeIaSED(ArnettDecaySED):
    r"""
    Type Ia supernova SED, with priors fit to a ZTF sample of SNe Ia.

    Reuses :class:`~uvex_transients.models.arnett.ArnettDecaySED`'s radioactive-decay Arnett diffusion and
    floored-photosphere blackbody entirely; only :attr:`_DEFAULT_PARAMETERS` differs. The priors follow
    :footcite:t:`2026arXiv260202677S` (Sarin+26, a sample of 2205 SNe Ia from ZTF):
    ``kappa_gamma`` is fixed at the value adopted by :footcite:t:`2014MNRAS.440.1498S` (Scalzo+14),
    and ``kappa`` is uniform over that paper's marginalization range.

    ``M_ej`` is a two-component Gaussian mixture rather than a single Gaussian,

    .. math::

        p(M_\mathrm{ej}) = f\,\mathcal{N}(\mu_1, \sigma_1) + (1 - f)\,\mathcal{N}(\mu_2, \sigma_2),

    reflecting a mix of sub-Chandrasekhar and near-Chandrasekhar explosions; earlier bolometric
    light-curve samples found a similar split (:footcite:t:`2014MNRAS.440.1498S`,
    :footcite:t:`2025arXiv250100638B`, Bora+24).

    .. warning::

        The mixture weight and component means/widths below are a by-eye starting point, not a fit;
        see the class notes in ``docs/source/transients/type_i.rst`` for the reasoning and refit
        before relying on this model quantitatively.

    .. rubric:: Parameters

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``M_Ni``
         - :math:`M_\mathrm{Ni}`
         - Nickel-56 mass synthesized in the explosion.
       * - ``M_ej``
         - :math:`M_\mathrm{ej}`
         - Ejecta mass; a two-component (sub-/near-Chandrasekhar) mixture.
       * - ``v_ej``
         - :math:`v_\mathrm{ej}`
         - Ejecta velocity, taken to be constant and equal to the photospheric velocity.
       * - ``kappa``
         - :math:`\kappa`
         - Grey optical opacity.
       * - ``kappa_gamma``
         - :math:`\kappa_\gamma`
         - Opacity to high-energy photons. Fixed.
       * - ``T_floor``
         - :math:`T_\mathrm{floor}`
         - Minimum photospheric temperature.

    References
    ----------
    .. footbibliography::
    """

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "M_Ni": Parameter(
            prior=TruncatedNormalPrior(mean=1.2, sigma=0.3, lower=0.05, upper=3.0),
            scale=1 * u.Msun,
            description="Nickel-56 mass synthesized in the explosion.",
            latex=r"M_\mathrm{Ni}",
        ),
        "M_ej": Parameter(
            prior=MixturePrior(
                components=(
                    TruncatedNormalPrior(mean=1.0, sigma=0.2, lower=0.05, upper=3.0),
                    TruncatedNormalPrior(mean=1.5, sigma=0.25, lower=0.05, upper=3.0),
                ),
                weights=np.array([0.3, 0.7]),
            ),
            scale=1 * u.Msun,
            description="Ejecta mass: a 0.3/0.7 mix of N(1.0, 0.2) sub-Chandrasekhar and "
            "N(1.5, 0.25) near-Chandrasekhar components (by-eye, not yet fit).",
            latex=r"M_\mathrm{ej}",
        ),
        "v_ej": Parameter(
            prior=NormalPrior(mean=11.0, sigma=1.0),
            scale=1e3 * u.km / u.s,
            description="Constant ejecta velocity, identified with the photospheric velocity.",
            latex=r"v_\mathrm{ej}",
        ),
        "kappa": Parameter(
            prior=UniformPrior(lower=0.05, upper=0.15),
            scale=1 * u.cm**2 / u.g,
            description="Grey optical opacity.",
            latex=r"\kappa",
        ),
        "kappa_gamma": Parameter(
            prior=ConstantPrior(value=0.03),
            scale=1 * u.cm**2 / u.g,
            description="Opacity to high-energy photons. Fixed (Scalzo+14).",
            latex=r"\kappa_\gamma",
        ),
        "T_floor": Parameter(
            prior=TruncatedNormalPrior(mean=6000.0, sigma=1000.0, lower=3000.0, upper=10000.0),
            scale=1 * u.K,
            description="Minimum photospheric temperature.",
            latex=r"T_\mathrm{floor}",
        ),
    }
