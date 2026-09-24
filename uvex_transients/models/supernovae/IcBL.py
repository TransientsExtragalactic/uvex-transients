"""Model of the SED from type Ic-BL (broad-lined, stripped-envelope) supernovae."""

from typing import ClassVar

from astropy import units as u

from uvex_transients.models.arnett import ArnettDecaySED
from uvex_transients.models.core.parameters import Parameter
from uvex_transients.models.core.priors import (
    ConstantPrior,
    TruncatedNormalPrior,
    UniformPrior,
)

__all__ = ["TypeIcBLSED"]


class TypeIcBLSED(ArnettDecaySED):
    r"""
    Type Ic-BL (broad-lined) supernova SED, with priors fit to a ZTF sample of SNe Ic-BL.

    Reuses :class:`~uvex_transients.models.arnett.ArnettDecaySED`'s radioactive-decay Arnett
    diffusion and floored-photosphere blackbody entirely, exactly as
    :class:`~uvex_transients.models.supernovae.Ia.TypeIaSED` does; only :attr:`_DEFAULT_PARAMETERS`
    differs. The priors on ``M_Ni``, ``M_ej`` and ``v_ej`` are sample statistics (mean, sample
    standard deviation) of the 36-event explosion-property table of :footcite:t:`srinivasaragavan2024`
    (nickel mass, kinetic energy, ejecta mass and photospheric velocity per event). One event
    (SN 2020wgz), whose reported
    :math:`M_\mathrm{Ni}=2.46\,M_\odot` is a >5-sigma outlier driven by an ``e_k``/``m_ej`` *lower
    limit* rather than a measurement, is excluded from the ``M_Ni`` statistics. ``M_ej`` uses only
    rows with a measured (non-lower-limit) value, and ``v_ej`` is identified with the sample's
    photospheric velocities (``v_ph``), regardless of the epoch quoted. ``kappa`` and ``T_floor``
    are not constrained by that table and are left at `TypeIaSED`'s values; ``kappa_gamma`` is
    instead fixed at a large value (full gamma-ray trapping across the simulated window), unlike
    `TypeIaSED`'s Scalzo+14 value.

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
         - Ejecta mass.
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
            prior=TruncatedNormalPrior(mean=0.33, sigma=0.24, lower=0.02, upper=2.0),
            scale=1 * u.Msun,
            description="Nickel-56 mass synthesized in the explosion.",
            latex=r"M_\mathrm{Ni}",
        ),
        "M_ej": Parameter(
            prior=TruncatedNormalPrior(mean=2.54, sigma=1.95, lower=0.1, upper=10.0),
            scale=1 * u.Msun,
            description="Ejecta mass.",
            latex=r"M_\mathrm{ej}",
        ),
        "v_ej": Parameter(
            prior=TruncatedNormalPrior(mean=20.1, sigma=4.96, lower=5.0, upper=45.0),
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
            prior=ConstantPrior(value=1000.0),
            scale=1 * u.cm**2 / u.g,
            description="Opacity to high-energy photons. Fixed at a large value (full gamma-ray trapping).",
            latex=r"\kappa_\gamma",
        ),
        "T_floor": Parameter(
            prior=TruncatedNormalPrior(mean=6000.0, sigma=1000.0, lower=3000.0, upper=10000.0),
            scale=1 * u.K,
            description="Minimum photospheric temperature.",
            latex=r"T_\mathrm{floor}",
        ),
    }
