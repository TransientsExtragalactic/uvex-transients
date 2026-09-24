"""Core-collapse supernova population."""

from abc import ABC
from typing import ClassVar, Union

import numpy as np
from astropy import units as u
from astropy.units import Quantity
from numpy.typing import NDArray

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

# Type IIP fraction of the total CC SNe rate (Li et al. 2011).
_TYPE_IIP_FRACTION = 0.40

# Early-interacting, IXF/GGI-like Type IIP SNe as a fraction of the ordinary Type IIP rate above,
# following the high incidence of early CSM-interaction signatures found among Type II SNe by
# Bruch et al. 2023 (ZTF).
_TYPE_IIP_EXCESS_FRACTION = 0.30 * _TYPE_IIP_FRACTION

# Type IIb fraction of the total CC SNe rate, Shivvers et al. 2017 (doi:10.1088/1538-3873/aa54a6).
_TYPE_IIB_FRACTION = 0.103

# Stripped-envelope (IIb + Ib + Ic) fraction of the total CC SNe rate, and the Type Ic and Type Ib
# shares of that stripped-envelope rate, all from Shivvers et al. 2017
# (doi:10.1088/1538-3873/aa54a6). The Ic and Ib fractions of the CC rate are the product.
_SESNE_FRACTION = 0.304
_TYPE_IC_FRACTION = _SESNE_FRACTION * 0.411
_TYPE_IB_FRACTION = _SESNE_FRACTION * 0.161

# Type I superluminous SNe (SLSNe-I) as a fraction of the total CC SNe rate: a local ratio of ~1/3500
# (+2800/-720), from the PTF rates of Frohmaier et al. 2021 (arXiv:2010.15270).
_SLSN_FRACTION = 1 / 3500

# None of the fractions above yet carry a published uncertainty of their own, so every subtype
# below leaves `RATE_CI` at its default (unset) for now.


class _CoreCollapseSNe(ExtragalacticTransient, ABC):
    """
    Shared machinery for every core-collapse SNe subtype below.

    Every subtype shares the same redshift shape (the Madau & Dickinson 2014 star-formation-history
    shape underlying `core_collapse_rate_shape`) and differs only in what fraction of the total
    core-collapse rate it represents (`RATE_FRACTION`) -- so `rate`/`rate_shape` are implemented
    here, once, in terms of that one class variable, rather than duplicated per subtype as they
    were before `ExtragalacticTransient.rate`/`rate_shape` existed.
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


class TypeIIPExcessSNe(_CoreCollapseSNe):
    """Early-interacting (IXF/GGI-like) Type IIP core-collapse SNe: `TypeIIPExcessSED`."""

    DEFAULT_MODEL = TypeIIPExcessSED
    DEFAULT_DURATION = 100 * u.day
    DEFAULT_Z_LIM = 1.2

    RATE_FRACTION = _TYPE_IIP_EXCESS_FRACTION


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


class TypeIbSNe(_CoreCollapseSNe):
    """Type Ib core-collapse SNe: `TypeIbSED` (single Bazin pulse x cooling blackbody)."""

    DEFAULT_MODEL = TypeIbSED
    DEFAULT_DURATION = 100 * u.day
    DEFAULT_Z_LIM = 0.5

    RATE_FRACTION = _TYPE_IB_FRACTION


class TypeIcSNe(_CoreCollapseSNe):
    """Type Ic core-collapse SNe: `TypeIcSED` (single Bazin pulse x cooling blackbody)."""

    DEFAULT_MODEL = TypeIcSED
    DEFAULT_DURATION = 100 * u.day
    DEFAULT_Z_LIM = 0.5

    RATE_FRACTION = _TYPE_IC_FRACTION


class MagnetarSLSNe(_CoreCollapseSNe):
    """
    Type I superluminous SNe powered by a magnetar spin-down engine: `ArnettMagnetarSpindownSED`.

    The SED's default priors are the sample-wide magnetar-model posteriors of Nicholl et al. 2017
    (38 SLSNe-I), so the population reproduces their peak luminosities (median ~3e44 erg/s, ~1e44--1e45
    at 1 sigma) and rest-frame rise times (median ~30 d). The rate is a fixed fraction of the
    core-collapse rate (~1/3500, Frohmaier et al. 2021), so it follows the star-formation history.

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

    @property
    def rate(self) -> Quantity:
        """~astropy.units.Quantity: The local (z=0) volumetric Type Ia rate (Maoz & Graur 2017 DTD x MD14 SFH)."""
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
