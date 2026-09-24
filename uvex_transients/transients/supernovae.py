"""Core-collapse supernova population."""

from abc import ABC
from typing import ClassVar, Union

import numpy as np
from astropy import units as u
from astropy.units import Quantity
from numpy.typing import NDArray

from uvex_transients.models._utils import convert_CI_to_fractional
from uvex_transients.models.supernovae import (
    ArnettMagnetarSpindownSED,
    MoragShockCoolingSED,
    TypeIaSED,
    TypeIbSED,
    TypeIcSED,
    TypeIIbSED,
    TypeIIPExcessSED,
    TypeIIPSED,
)
from uvex_transients.utils.cosmology import (
    core_collapse_rate_coefficient,
    core_collapse_rate_shape,
    supernovae_Ia_rate,
)

from .base import ExtragalacticTransient

# ============================================== #
# Core-Collapse Rate Demographics                #
# ============================================== #
# Every subtype's `RATE_FRACTION` below is a *chain* of independently measured fractions-of-a-
# parent-rate, e.g. "Type Ib is 35.6% of the stripped-envelope rate, which is itself 30.4% of the
# total CC rate" -- so each link in that chain is stored here as measured (a fraction plus its own
# published uncertainty), and `RATE_FRACTION`/`RATE_CI` are built by multiplying the relevant links
# and propagating their fractional uncertainties, rather than hand-computing and hardcoding the
# already-multiplied-out numbers. `convert_CI_to_fractional` turns each literature-quoted "value
# +upper/-lower" into a `(lower, upper)` fractional-error pair; `_combine_fractional_errors`
# propagates any number of *independent* such pairs through a product, in quadrature (standard
# linear/small-error propagation in log space); `_fractional_error_to_rate_ci` turns a combined
# fractional error into the multiplicative `RATE_CI` form every `ExtragalacticTransient` uses.


def _combine_fractional_errors(*errors: tuple[float, float]) -> tuple[float, float]:
    r"""
    Combine independent fractional ``(lower, upper)`` errors in quadrature.

    For independent quantities :math:`x_i`, each with its own fractional error
    :math:`(\delta_i^-, \delta_i^+)` (e.g. as returned by `convert_CI_to_fractional`), the
    fractional error of their product :math:`\prod_i x_i` is
    :math:`\bigl(\sqrt{\sum_i(\delta_i^-)^2},\ \sqrt{\sum_i(\delta_i^+)^2}\bigr)` -- ordinary
    linear error propagation applied to :math:`\ln\prod_i x_i = \sum_i \ln x_i`.

    Parameters
    ----------
    *errors : tuple of float
        Any number of ``(lower, upper)`` fractional-error pairs for independent factors in a
        product.

    Returns
    -------
    tuple of float
        The combined ``(lower, upper)`` fractional error of the product.
    """
    lower = float(np.sqrt(sum(lo**2 for lo, _ in errors)))
    upper = float(np.sqrt(sum(hi**2 for _, hi in errors)))
    return lower, upper


def _fractional_error_to_rate_ci(error: tuple[float, float]) -> tuple[float, float]:
    """Convert a ``(lower, upper)`` fractional error into `RATE_CI`-style multiplicative bounds."""
    lower, upper = error
    return 1.0 - lower, 1.0 + upper


# ------------------------------------------------------------------------------------------- #
# Shivvers et al. 2017 (doi:10.1088/1538-3873/aa54a6): split of the total CC SNe rate into
# H-rich Type II and stripped-envelope (SESNe) events, then the SESNe rate into IIb/Ib/Ic/Ic-BL.
# ------------------------------------------------------------------------------------------- #
_TYPE_II_OF_CC = 0.696
_TYPE_II_OF_CC_ERR = convert_CI_to_fractional(_TYPE_II_OF_CC, _TYPE_II_OF_CC + 0.067, _TYPE_II_OF_CC - 0.067)

_SESNE_FRACTION = 0.304  # Stripped-envelope (IIb + Ib + Ic [+ Ic-BL]) fraction of the total CC rate.
_SESNE_FRACTION_ERR = convert_CI_to_fractional(_SESNE_FRACTION, _SESNE_FRACTION + 0.050, _SESNE_FRACTION - 0.049)

_IIB_OF_SESNE = 0.340
_IIB_OF_SESNE_ERR = convert_CI_to_fractional(_IIB_OF_SESNE, _IIB_OF_SESNE + 0.111, _IIB_OF_SESNE - 0.111)

_IB_OF_SESNE = 0.356
_IB_OF_SESNE_ERR = convert_CI_to_fractional(_IB_OF_SESNE, _IB_OF_SESNE + 0.114, _IB_OF_SESNE - 0.114)

_IC_OF_SESNE = 0.215
_IC_OF_SESNE_ERR = convert_CI_to_fractional(_IC_OF_SESNE, _IC_OF_SESNE + 0.086, _IC_OF_SESNE - 0.086)

# Ic-BL is not (yet) its own `_CoreCollapseSNe` subtype, so it has no `RATE_FRACTION`/`RATE_CI` of
# its own below -- stored here only so Shivvers et al. 2017's full stripped-envelope demographic
# breakdown is documented together in one place.
_ICBL_OF_SESNE = 0.037
_ICBL_OF_SESNE_ERR = convert_CI_to_fractional(_ICBL_OF_SESNE, _ICBL_OF_SESNE + 0.029, _ICBL_OF_SESNE - 0.037)

# ------------------------------------------------------------------------------------------- #
# Li et al. 2011 (doi:10.1111/j.1365-2966.2011.18160.x): split of the Type II rate into II-L and
# II-P, measured relative to the *Type II* rate above (not directly to the total CC rate).
# ------------------------------------------------------------------------------------------- #
# II-L is not (yet) its own `_CoreCollapseSNe` subtype; stored for the same reason as Ic-BL above.
_IIL_OF_TYPEII = 0.097
_IIL_OF_TYPEII_ERR = convert_CI_to_fractional(_IIL_OF_TYPEII, _IIL_OF_TYPEII + 0.040, _IIL_OF_TYPEII - 0.032)

_IIP_OF_TYPEII = 0.699
_IIP_OF_TYPEII_ERR = convert_CI_to_fractional(_IIP_OF_TYPEII, _IIP_OF_TYPEII + 0.051, _IIP_OF_TYPEII - 0.058)

# ------------------------------------------------------------------------------------------- #
# Strolger et al. 2015 (doi:10.1088/0004-637X/813/2/93): the core-collapse efficiency itself,
# k_CC = 0.0070 SNe per solar mass formed (+27%/-31%). This is `_CORE_COLLAPSE_EFFICIENCY` in
# `uvex_transients.utils.cosmology` (whose point estimate is left there, alongside the rest of the
# star-formation-history machinery it's used with); its *uncertainty* is shared by every
# core-collapse subtype below (they all multiply the same `core_collapse_rate_coefficient`), so it
# is combined into each subtype's own fraction-of-parent uncertainty here, once.
# ------------------------------------------------------------------------------------------- #
_CC_NORMALIZATION_ERR = (0.31, 0.27)


def _with_cc_normalization_uncertainty(fraction_err: tuple[float, float]) -> tuple[float, float]:
    r"""
    Combine a subtype's own fraction-of-parent error with the core-collapse normalization uncertainty.

    The latter is the shared Strolger et al. 2015 uncertainty on :math:`k_\mathrm{CC}`
    (`_CC_NORMALIZATION_ERR`), common to every `_CoreCollapseSNe` subtype. Returns `RATE_CI`-style
    multiplicative bounds.
    """
    return _fractional_error_to_rate_ci(_combine_fractional_errors(fraction_err, _CC_NORMALIZATION_ERR))


# ------------------------------------------------------------------------------------------- #
# RATE_FRACTION / RATE_CI for each modeled subtype, built from the measured fractions above.
# ------------------------------------------------------------------------------------------- #
# Type IIP fraction of the total CC rate: Li et al. 2011's II-P share of the Type II rate, times
# Shivvers et al. 2017's Type II share of the total CC rate.
_TYPE_IIP_FRACTION = _TYPE_II_OF_CC * _IIP_OF_TYPEII
_TYPE_IIP_FRACTION_ERR = _combine_fractional_errors(_TYPE_II_OF_CC_ERR, _IIP_OF_TYPEII_ERR)

# Early-interacting, IXF/GGI-like Type IIP SNe as a fraction of the ordinary Type IIP rate above,
# following the high incidence of early CSM-interaction signatures found among Type II SNe by
# Bruch et al. 2023 (ZTF). This 30% multiplier has no published uncertainty of its own, so it
# doesn't change the fractional error inherited from `_TYPE_IIP_FRACTION_ERR`.
_TYPE_IIP_EXCESS_FRACTION = 0.30 * _TYPE_IIP_FRACTION
_TYPE_IIP_EXCESS_FRACTION_ERR = _TYPE_IIP_FRACTION_ERR

_TYPE_IIB_FRACTION = _SESNE_FRACTION * _IIB_OF_SESNE
_TYPE_IIB_FRACTION_ERR = _combine_fractional_errors(_SESNE_FRACTION_ERR, _IIB_OF_SESNE_ERR)

_TYPE_IB_FRACTION = _SESNE_FRACTION * _IB_OF_SESNE
_TYPE_IB_FRACTION_ERR = _combine_fractional_errors(_SESNE_FRACTION_ERR, _IB_OF_SESNE_ERR)

_TYPE_IC_FRACTION = _SESNE_FRACTION * _IC_OF_SESNE
_TYPE_IC_FRACTION_ERR = _combine_fractional_errors(_SESNE_FRACTION_ERR, _IC_OF_SESNE_ERR)

# Type I superluminous SNe (SLSNe-I) as a fraction of the total CC SNe rate: a local ratio of
# 1/3500 (+2800/-720 on the *denominator*), from the PTF rates of Frohmaier et al. 2021
# (arXiv:2010.15270), i.e. the rate itself spans 1/6300 to 1/2780.
_SLSN_FRACTION = 1 / 3500
_SLSN_FRACTION_ERR = convert_CI_to_fractional(_SLSN_FRACTION, 1 / (3500 - 720), 1 / (3500 + 2800))


class _CoreCollapseSNe(ExtragalacticTransient, ABC):
    r"""
    Shared machinery for every core-collapse SNe subtype below.

    Every subtype shares the same redshift shape (the Madau & Dickinson 2014 star-formation-history
    shape underlying `core_collapse_rate_shape`) and differs only in what fraction of the total
    core-collapse rate it represents (`RATE_FRACTION`) -- so `rate`/`rate_shape` are implemented
    here, once, in terms of that one class variable, rather than duplicated per subtype as they
    were before `ExtragalacticTransient.rate`/`rate_shape` existed.

    Each subtype's `RATE_CI` also folds in the same shared core-collapse normalization uncertainty
    (Strolger et al. 2015's :math:`k_\mathrm{CC}`, :math:`+27\%/-31\%`), combined in quadrature with
    that subtype's own fraction-of-parent uncertainty via `_with_cc_normalization_uncertainty` --
    see the module-level comments above for the full demographic breakdown this is built from.
    """

    RATE_FRACTION: ClassVar[float | None] = None
    """float: This subtype's fraction of the total core-collapse SNe rate. Must be set by subclasses."""

    @property
    def rate(self) -> Quantity:
        """~astropy.units.Quantity: `RATE_FRACTION` of the total core-collapse rate, at this instance's `cosmology`."""
        return self.RATE_FRACTION * core_collapse_rate_coefficient(self.cosmology)

    def rate_shape(self, z: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
        """
        Return the shared core-collapse SNe redshift shape (Madau & Dickinson 2014) at redshift(s) `z`.

        Parameters
        ----------
        z : float or numpy.ndarray
            Redshift(s) at which to evaluate the rate shape.

        Returns
        -------
        float or numpy.ndarray
            The dimensionless rate shape, common to every core-collapse SNe subtype.
        """
        return core_collapse_rate_shape(z)


class TypeIIPSNe(_CoreCollapseSNe):
    """Type IIP core-collapse SNe: `TypeIIPSED` (two-exponential + radioactive-tail lightcurve x cooling blackbody)."""

    DEFAULT_MODEL = TypeIIPSED
    DEFAULT_DURATION = 100 * u.day
    DEFAULT_Z_LIM = 0.8

    RATE_FRACTION = _TYPE_IIP_FRACTION
    RATE_CI = _with_cc_normalization_uncertainty(_TYPE_IIP_FRACTION_ERR)


class TypeIIPExcessSNe(_CoreCollapseSNe):
    """Early-interacting (IXF/GGI-like) Type IIP core-collapse SNe: `TypeIIPExcessSED`."""

    DEFAULT_MODEL = TypeIIPExcessSED
    DEFAULT_DURATION = 100 * u.day
    DEFAULT_Z_LIM = 1.2

    RATE_FRACTION = _TYPE_IIP_EXCESS_FRACTION
    RATE_CI = _with_cc_normalization_uncertainty(_TYPE_IIP_EXCESS_FRACTION_ERR)


class ShockCoolingIIb(_CoreCollapseSNe):
    """
    Early-time shock-cooling emission from Type IIb core-collapse SNe.

    Uses `MoragShockCoolingSED` (Morag et al. 2024).

    This models only the shock-cooling phase -- from shock breakout out to roughly a week
    post-explosion, per Morag+24's own stated validity window -- not the radioactive-decay-powered
    peak that dominates a typical Type IIb light curve at ~15-25 days. `DEFAULT_DURATION` is set
    accordingly, well short of a full Type IIb light curve.
    """

    DEFAULT_MODEL = MoragShockCoolingSED
    DEFAULT_DURATION = 20 * u.day
    DEFAULT_Z_LIM = 1

    RATE_FRACTION = _TYPE_IIB_FRACTION
    RATE_CI = _with_cc_normalization_uncertainty(_TYPE_IIB_FRACTION_ERR)


class TypeIIbSNe(_CoreCollapseSNe):
    """
    Full Type IIb core-collapse SN light curve.

    `TypeIIbSED`: a phenomenological superposition of two Bazin pulses (an early shock-cooling
    peak and the radioactively powered main peak) times a cooling blackbody photosphere. A broad
    prior on the early peak's amplitude covers both single- and double-peaked Type IIb events in
    the same population.

    Unlike `ShockCoolingIIb`, which covers only the early shock-cooling phase, this spans a full
    Type IIb light curve, hence its much longer `DEFAULT_DURATION`. Shares the same event rate as
    `ShockCoolingIIb` -- both describe the same underlying Type IIb population, just with
    different SED models.
    """

    DEFAULT_MODEL = TypeIIbSED
    DEFAULT_DURATION = 200 * u.day
    DEFAULT_Z_LIM = 0.5

    RATE_FRACTION = _TYPE_IIB_FRACTION
    RATE_CI = _with_cc_normalization_uncertainty(_TYPE_IIB_FRACTION_ERR)


class TypeIbSNe(_CoreCollapseSNe):
    """Type Ib core-collapse SNe: `TypeIbSED` (single Bazin pulse x cooling blackbody)."""

    DEFAULT_MODEL = TypeIbSED
    DEFAULT_DURATION = 100 * u.day
    DEFAULT_Z_LIM = 0.5

    RATE_FRACTION = _TYPE_IB_FRACTION
    RATE_CI = _with_cc_normalization_uncertainty(_TYPE_IB_FRACTION_ERR)


class TypeIcSNe(_CoreCollapseSNe):
    """Type Ic core-collapse SNe: `TypeIcSED` (single Bazin pulse x cooling blackbody)."""

    DEFAULT_MODEL = TypeIcSED
    DEFAULT_DURATION = 100 * u.day
    DEFAULT_Z_LIM = 0.5

    RATE_FRACTION = _TYPE_IC_FRACTION
    RATE_CI = _with_cc_normalization_uncertainty(_TYPE_IC_FRACTION_ERR)


class MagnetarSLSNe(_CoreCollapseSNe):
    """
    Type I superluminous SNe powered by a magnetar spin-down engine: `ArnettMagnetarSpindownSED`.

    The SED's default priors are the sample-wide magnetar-model posteriors of Nicholl et al. 2017
    (38 SLSNe-I), so the population reproduces their peak luminosities (median ~3e44 erg/s, ~1e44--1e45
    at 1 sigma) and rest-frame rise times (median ~30 d). The rate is a fixed fraction of the
    core-collapse rate (1/3500, +2800/-720 on the denominator; Frohmaier et al. 2021), so it
    follows the star-formation history.

    Events are long-lived: the bolometric light curve falls to 1e-3 of peak after ~250--1400 d
    (16th--84th percentile of the prior, rest frame, median ~540 d). `DEFAULT_DURATION` covers the
    bright part of the decline for most events, but the slowest ones are truncated by the window and
    their faint tails are not simulated.

    `DEFAULT_Z_LIM` is set from an actual `generate_events`/`filter_by_snr` run against the default
    schedule (25 AB mag limiting-magnitude screen, SNR > 5): out to z = 8, 99.7% of detected events fall
    below z = 4, and the detected count per redshift bin is already declining by z ~ 1.5, well inside that.
    A lower `DEFAULT_Z_LIM` (e.g. the earlier value of 2) truncates a real, UV-bright, high-redshift tail
    rather than one outside UVEX's reach.
    """

    DEFAULT_MODEL = ArnettMagnetarSpindownSED
    DEFAULT_DURATION = 600 * u.day
    DEFAULT_Z_LIM = 4

    RATE_FRACTION = _SLSN_FRACTION
    RATE_CI = _with_cc_normalization_uncertainty(_SLSN_FRACTION_ERR)


# ============================================== #
# Type Ia Rate Demographics                      #
# ============================================== #
# Maoz & Graur 2017 (:footcite:t:`2017ApJ...848...25M`): N_Ia/M_star = (1.3 +/- 0.1) x 10^-3
# Msun^-1, the normalization of their power-law delay-time distribution -- `_MAOZ_GRAUR_NIA_PER_M`
# in `uvex_transients.utils.cosmology`, whose point estimate is left there alongside the DTD
# machinery it belongs to.
_MAOZ_GRAUR_NIA_PER_M = 1.3e-3
_MAOZ_GRAUR_NIA_PER_M_SIGMA = 0.1e-3

# `supernovae_Ia_rate` is exactly linear in `n_ia_per_m` (it enters only as an overall
# multiplicative scale on the DTD amplitude -- see `_maoz_graur_norm`), so rather than assuming
# that analytically, `n_ia_per_m`'s uncertainty is propagated by calling `supernovae_Ia_rate`
# itself at its +/-1 sigma endpoints and taking the ratio to the fiducial rate. This has been
# checked to be exactly cosmology-independent (as expected from the linearity above), so the one
# ratio computed here at the default cosmology applies unchanged regardless of `self.cosmology`.
_TYPE_IA_FIDUCIAL_RATE = supernovae_Ia_rate(0.0, n_ia_per_m=_MAOZ_GRAUR_NIA_PER_M)
_TYPE_IA_RATE_CI: tuple[float, float] = (
    float(
        (
            supernovae_Ia_rate(0.0, n_ia_per_m=_MAOZ_GRAUR_NIA_PER_M - _MAOZ_GRAUR_NIA_PER_M_SIGMA)
            / _TYPE_IA_FIDUCIAL_RATE
        ).to_value(u.dimensionless_unscaled)
    ),
    float(
        (
            supernovae_Ia_rate(0.0, n_ia_per_m=_MAOZ_GRAUR_NIA_PER_M + _MAOZ_GRAUR_NIA_PER_M_SIGMA)
            / _TYPE_IA_FIDUCIAL_RATE
        ).to_value(u.dimensionless_unscaled)
    ),
)


class TypeIaSNe(ExtragalacticTransient):
    """
    Type Ia (thermonuclear) supernova population.

    Modeled with `TypeIaSED` (an Arnett-style radioactive-decay diffusion light curve with a
    floored-photosphere blackbody photosphere, its priors fit to the ZTF SNe Ia sample of
    Sarin et al. 2026). Unlike the `_CoreCollapseSNe` subtypes above, SNe Ia are thermonuclear
    rather than core-collapse events, so this class does not share their machinery: the rate is
    the cosmic star formation history convolved with the Maoz & Graur (2017) power-law delay-time
    distribution (`supernovae_Ia_rate`), not a fixed fraction of the core-collapse rate. The delay
    times are broadly distributed (a power law from 40 Myr to the age of the universe), so the
    resulting rate shape tracks the *integrated* star-formation history rather than the
    core-collapse subtypes' instantaneous one -- flatter at low z and slower to decline at high z.

    `DEFAULT_DURATION` (365 d) covers the rise to peak (median ~14 d after explosion, in this
    model's prior) through the decline to 1e-3 of peak for nearly the whole prior (16th--84th
    percentile ~270--325 d, rest frame).

    `DEFAULT_Z_LIM` is set from an actual `sample_event_redshift`/peak-apparent-magnitude check
    against the UVEX bandpasses (25 AB mag limiting-magnitude screen): with `redshift_limit`
    temporarily raised to 4, no simulated event peaks above the limit beyond z ~ 0.8 in either
    band, and the NUV-detected fraction per redshift bin has already fallen to zero by z = 1 --
    consistent with this model's fixed, non-evolving ``kappa_gamma`` leaving no UV-bright
    high-redshift tail the way `MagnetarSLSNe`'s magnetar engine does.
    """

    DEFAULT_MODEL = TypeIaSED
    DEFAULT_DURATION = 365 * u.day
    DEFAULT_Z_LIM = 1.0

    RATE_CI = _TYPE_IA_RATE_CI

    @property
    def rate(self) -> Quantity:
        """
        ~astropy.units.Quantity: The local (z=0) volumetric Type Ia rate (Maoz & Graur 2017 DTD x MD14 SFH).

        See Also
        --------
        RATE_CI : The confidence bounds attached to this point estimate.
        rate_shape : The redshift dependence this rate normalizes.
        """
        return supernovae_Ia_rate(0.0, cosmology=self.cosmology)

    def rate_shape(self, z: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
        """
        Return the DTD-convolved Type Ia rate shape (Maoz & Graur 2017 x Madau & Dickinson 2014) at `z`.

        Parameters
        ----------
        z : float or numpy.ndarray
            Redshift(s) at which to evaluate the rate shape.

        Returns
        -------
        float or numpy.ndarray
            `supernovae_Ia_rate(z)`, normalized by `rate` so that the shape equals 1 at ``z=0``.
        """
        z_arr = np.asarray(z, dtype=np.float64)
        numerator = np.atleast_1d(supernovae_Ia_rate(z_arr, cosmology=self.cosmology).to_value(u.Mpc**-3 * u.yr**-1))
        denominator = self.rate.to_value(u.Mpc**-3 * u.yr**-1)
        shape = numerator / denominator
        return shape if z_arr.ndim > 0 else shape.item()
