r"""
Blackbody spectral shape.

A :class:`BlackbodySpectrum` is shaped entirely by a single sampled
``temperature`` -- unlike a power law, which is scale-free and needs an
arbitrary reference frequency to anchor it, a blackbody's shape has no free
normalization choice: the Planck function, normalized by the
Stefan-Boltzmann law, already integrates to exactly 1 over all frequencies
for any temperature.

:class:`ModifiedBlackbodySpectrum` multiplies that shape by a smooth suppression
blueward of a break frequency, mimicking the UV line blanketing of supernova photospheres
(Nicholl et al. 2017). It is no longer normalized by construction, so its normalization is
the Stefan-Boltzmann-scaled integral of the suppressed Planck function, looked up from an
interpolation table (see
:func:`~uvex_transients.models._util_functions.modified_blackbody_norm_interpolator`).
"""

from functools import cache
from typing import ClassVar

import numpy as np
from astropy import units as u

from uvex_transients.models._constants import H_CGS, K_B_CGS, SIGMA_SB_CGS
from uvex_transients.models._typing import CGSParameterValue, FloatArray
from uvex_transients.models.core.base import Spectrum
from uvex_transients.models.core.parameters import Parameter
from uvex_transients.models.core.priors import ConstantPrior, LogNormalPrior

from .._util_functions import (
    modified_blackbody_intensity_log_cgs,
    modified_blackbody_norm_interpolator,
    planck_shape_log_cgs,
)

__all__ = ["BlackbodySpectrum", "ModifiedBlackbodySpectrum"]


class BlackbodySpectrum(Spectrum):
    r"""
    Blackbody spectral shape, shaped entirely by a sampled ``temperature``.

    Represents :math:`S(\nu, T) = \pi B_\nu(\nu, T) / (\sigma T^4)`, the
    Lambertian-emergent Planck function normalized by the Stefan-Boltzmann
    constant so that :math:`\int_0^\infty S(\nu, T)\,d\nu = 1` for any
    ``temperature`` -- see the module docstring for why this needs no
    reference-frequency pivot the way a power law does.

    By default ``temperature`` uses a
    :class:`~uvex_transients.models.core.priors.LogNormalPrior`, a physically motivated
    prior for a strictly positive scale parameter; like any other
    :class:`~uvex_transients.models.core.parameters.Parameter`, it can be overridden
    per instance.

    .. rubric:: Parameters

    The spectral shape parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``temperature``
         - :math:`T`
         - Blackbody temperature.
    """

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "temperature": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e4 * u.K,
            description="Blackbody temperature.",
            latex=r"T",
        ),
    }

    # ----------------------------------- #
    # Shape: S(nu)                        #
    # ----------------------------------- #
    # Narrowing `**parameters` to this model's own named parameters (rather
    # than matching the base class's generic `**parameters` exactly) is the
    # intended pattern for every `_eval` override -- see `Spectrum._eval`.
    @classmethod
    def _eval(  # type: ignore[override]
        cls, nu: FloatArray, *, temperature: CGSParameterValue
    ) -> FloatArray:
        return planck_shape_log_cgs(nu, temperature)

    # ----------------------------------- #
    # Normalization: integral of S(nu) dnu #
    # ----------------------------------- #
    @classmethod
    def _eval_normalization(cls, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        Natural log of the shape integral.

        Exactly 0 (i.e. the integral itself is 1) for any ``temperature``, by
        construction -- see the class docstring.
        """
        return np.zeros_like(np.asarray(parameters["temperature"], dtype=np.float64))


@cache
def _norm_interpolator(smoothing: float):
    """Build (once per ``smoothing``) the interpolated suppressed-Planck integral over ``(x_0, beta)``."""
    return modified_blackbody_norm_interpolator(
        smoothing, x_0_range=(1e-2, 1e3), beta_range=(0.0, 10.0), n_x_0=400, n_beta=81, n_x=1000
    )


class ModifiedBlackbodySpectrum(Spectrum):
    r"""
    Line-blanketed blackbody spectral shape.

    Represents the Stefan-Boltzmann-normalized Planck shape of :class:`BlackbodySpectrum`,
    multiplied by a smooth suppression blueward of ``break_frequency`` (Nicholl et al. 2017),

    .. math::

        S(\nu) = \frac{\pi B_\nu(\nu, T)}{\sigma T^4}
        \left[1 + \left(\frac{\nu}{\nu_\mathrm{break}}\right)^{s}\right]^{-\beta/s},

    which falls as :math:`\nu^{-\beta}` well above the break. The smoothing :math:`s` is fixed
    at :attr:`SMOOTHING`. Unlike :class:`BlackbodySpectrum`, the suppression removes flux, so
    :math:`\int S\,d\nu < 1` and the normalization depends on ``temperature``,
    ``break_frequency`` and ``beta``. It equals :math:`(15/\pi^4)\,I(x_0, \beta)` with
    :math:`x_0 = h\nu_\mathrm{break}/k_BT`, where :math:`I` is evaluated by
    :func:`~uvex_transients.models._util_functions.normalize_modified_blackbody` and looked up
    from an interpolation table. Parameters outside the table's range
    (:math:`10^{-2} \le x_0 \le 10^3`, :math:`0 \le \beta \le 10`) raise a :class:`ValueError`;
    inside it the table is accurate to better than 1%.

    By default ``temperature`` uses a
    :class:`~uvex_transients.models.core.priors.LogNormalPrior`, while ``break_frequency``
    (3000 Angstrom) and ``beta`` use constant priors, making them deterministic unless replaced
    or overridden per instance.

    .. rubric:: Parameters

    The spectral shape parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``temperature``
         - :math:`T`
         - Blackbody temperature.
       * - ``break_frequency``
         - :math:`\nu_\mathrm{break}`
         - Frequency below which the blanketing suppression sets in (generally in the UV).
       * - ``beta``
         - :math:`\beta`
         - Power-law index of the suppression above the break.
    """

    SMOOTHING: ClassVar[float] = 10.0
    """float: The smoothing parameter :math:`s` of the break, fixed for every instance."""

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "temperature": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e4 * u.K,
            description="Blackbody temperature.",
            latex=r"T",
        ),
        "break_frequency": Parameter(
            prior=ConstantPrior(value=1.0),
            scale=1e15 * u.Hz,
            description="Frequency above which the line-blanketing suppression sets in.",
            latex=r"\nu_{\rm break}",
        ),
        "beta": Parameter(
            prior=ConstantPrior(value=1.0),
            scale=1.0,
            description="Power-law index of the suppression above the break.",
            latex=r"\beta",
        ),
    }

    # ----------------------------------- #
    # Shape: S(nu)                        #
    # ----------------------------------- #
    @classmethod
    def _eval(  # type: ignore[override]
        cls,
        nu: FloatArray,
        *,
        temperature: CGSParameterValue,
        break_frequency: CGSParameterValue,
        beta: CGSParameterValue,
    ) -> FloatArray:
        # pi / (sigma T^4) turns the Planck intensity into the unit-integral shape.
        norm = np.pi / (SIGMA_SB_CGS * np.asarray(temperature, dtype=np.float64) ** 4)
        return modified_blackbody_intensity_log_cgs(nu, temperature, break_frequency, beta, norm, cls.SMOOTHING)

    # ----------------------------------- #
    # Normalization: integral of S(nu) dnu #
    # ----------------------------------- #
    @classmethod
    def _eval_normalization(cls, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        Natural log of the shape integral, :math:`\ln[(15/\pi^4)\,I(x_0, \beta)]`.

        The :math:`15/\pi^4` factor is the Stefan-Boltzmann normalization, so that the
        unsuppressed integral is exactly 1.
        """
        temperature = np.asarray(parameters["temperature"], dtype=np.float64)
        x_0 = H_CGS * np.asarray(parameters["break_frequency"], dtype=np.float64) / (K_B_CGS * temperature)
        integral = _norm_interpolator(cls.SMOOTHING)(x_0, parameters["beta"])
        return np.log(15.0 / np.pi**4 * integral)
