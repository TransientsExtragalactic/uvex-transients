"""Core-collapse supernova population."""

from typing import Union

import numpy as np
from astropy import units as u
from numpy.typing import NDArray

from uvex_transients.models.supernovae import (
    ArnettMagnetarSpindownSED,
    MoragShockCoolingSED,
    TypeIbSED,
    TypeIcSED,
    TypeIIbSED,
    TypeIIPExcessSED,
    TypeIIPSED,
)
from uvex_transients.utils.cosmology import core_collapse_rate

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


class TypeIIPSNe(ExtragalacticTransient):
    """Type IIP core-collapse SNe: `TypeIIPSED` (two-exponential + radioactive-tail lightcurve x cooling blackbody)."""

    DEFAULT_MODEL = TypeIIPSED
    DEFAULT_DURATION = 100 * u.day
    DEFAULT_Z_LIM = 0.8

    def event_rate(self, z: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
        """
        Return the volumetric Type IIP supernova rate.

        Parameters
        ----------
        z : float or numpy.ndarray
            Redshift at which to evaluate the event rate.

        Returns
        -------
        float or numpy.ndarray
            Type IIP volumetric event rate, given by the core-collapse
            supernova rate multiplied by the adopted Type IIP fraction.
        """
        return _TYPE_IIP_FRACTION * core_collapse_rate(z, cosmology=self.cosmology)


class TypeIIPExcessSNe(ExtragalacticTransient):
    """Early-interacting (IXF/GGI-like) Type IIP core-collapse SNe: `TypeIIPExcessSED`."""

    DEFAULT_MODEL = TypeIIPExcessSED
    DEFAULT_DURATION = 100 * u.day
    DEFAULT_Z_LIM = 1.2

    def event_rate(self, z: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
        """
        Return the volumetric early-interacting Type IIP supernova rate.

        Parameters
        ----------
        z : float or numpy.ndarray
            Redshift at which to evaluate the event rate.

        Returns
        -------
        float or numpy.ndarray
            Early-interacting Type IIP volumetric event rate, given by the
            core-collapse supernova rate multiplied by the adopted Type IIP
            excess fraction.
        """
        return _TYPE_IIP_EXCESS_FRACTION * core_collapse_rate(z, cosmology=self.cosmology)


class ShockCoolingIIb(ExtragalacticTransient):
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

    def event_rate(self, z: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
        """
        Return the volumetric Type IIb supernova rate.

        Parameters
        ----------
        z : float or numpy.ndarray
            Redshift at which to evaluate the event rate.

        Returns
        -------
        float or numpy.ndarray
            Type IIb volumetric event rate, given by the core-collapse
            supernova rate multiplied by the adopted Type IIb fraction.
        """
        return _TYPE_IIB_FRACTION * core_collapse_rate(z, cosmology=self.cosmology)


class TypeIIbSNe(ExtragalacticTransient):
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

    def event_rate(self, z: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
        """
        Return the volumetric Type Ib supernova rate.

        Parameters
        ----------
        z : float or numpy.ndarray
            Redshift at which to evaluate the event rate.

        Returns
        -------
        float or numpy.ndarray
            Type Ib volumetric event rate, given by the core-collapse
            supernova rate multiplied by the adopted Type Ib fraction.
        """
        return _TYPE_IIB_FRACTION * core_collapse_rate(z, cosmology=self.cosmology)


class TypeIbSNe(ExtragalacticTransient):
    """Type Ib core-collapse SNe: `TypeIbSED` (single Bazin pulse x cooling blackbody)."""

    DEFAULT_MODEL = TypeIbSED
    DEFAULT_DURATION = 100 * u.day
    DEFAULT_Z_LIM = 0.5

    def event_rate(self, z: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
        """
        Return the volumetric Type Ib supernova rate.

        Parameters
        ----------
        z : float or numpy.ndarray
            Redshift at which to evaluate the event rate.

        Returns
        -------
        float or numpy.ndarray
            Type Ib volumetric event rate, given by the core-collapse
            supernova rate multiplied by the adopted Type Ib fraction.
        """
        return _TYPE_IB_FRACTION * core_collapse_rate(z, cosmology=self.cosmology)


class TypeIcSNe(ExtragalacticTransient):
    """Type Ic core-collapse SNe: `TypeIcSED` (single Bazin pulse x cooling blackbody)."""

    DEFAULT_MODEL = TypeIcSED
    DEFAULT_DURATION = 100 * u.day
    DEFAULT_Z_LIM = 0.5

    def event_rate(self, z: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
        """
        Return the volumetric Type Ic supernova rate.

        Parameters
        ----------
        z : float or numpy.ndarray
            Redshift at which to evaluate the event rate.

        Returns
        -------
        float or numpy.ndarray
            Type Ic volumetric event rate, given by the core-collapse
            supernova rate multiplied by the adopted Type Ic fraction.
        """
        return _TYPE_IC_FRACTION * core_collapse_rate(z, cosmology=self.cosmology)


class MagnetarSLSNe(ExtragalacticTransient):
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

    def event_rate(self, z: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
        """
        Return the volumetric Type I superluminous supernova rate.

        Parameters
        ----------
        z : float or numpy.ndarray
            Redshift at which to evaluate the event rate.

        Returns
        -------
        float or numpy.ndarray
            Type I superluminous supernova volumetric event rate, given by
            the core-collapse supernova rate multiplied by the adopted
            SLSN-I fraction.
        """
        return _SLSN_FRACTION * core_collapse_rate(z, cosmology=self.cosmology)
