"""Core-collapse supernova population."""

from typing import Union

import numpy as np
from astropy import units as u
from numpy.typing import NDArray

from uvex_transients.models.supernovae import TypeIIPExcessSED, TypeIIPSED

from .base import ExtragalacticTransient

# Type IIP fraction of the total CC SNe rate (Li et al. 2011).
_TYPE_IIP_FRACTION = 0.40

# Early-interacting, IXF/GGI-like Type IIP SNe as a fraction of the ordinary Type IIP rate above,
# following the high incidence of early CSM-interaction signatures found among Type II SNe by
# Bruch et al. 2023 (ZTF).
_TYPE_IIP_EXCESS_FRACTION = 0.30 * _TYPE_IIP_FRACTION


def _core_collapse_rate(z: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
    """
    Return the total (all-subtype) volumetric core-collapse SNe rate at redshift(s) `z`.

    Shared by each subtype class's own `event_rate` below, so the per-subtype rates
    always stay a fixed fraction of the same underlying total rather than risking
    independent drift.

    Parameters
    ----------
    z : float or array-like
        Redshift(s) at which to evaluate the event rate.

    Returns
    -------
    float or array-like
        The volumetric event rate at the specified redshift(s), in events per cubic megaparsec per year.
    """
    z = np.asarray(z)

    # Maude & Dickenson coefficient * CC rate from LGS 2015.
    _coefficient = 0.0001365 * 0.49  # (0.49 from h^2 inclusion). TODO: We should be more robust.
    rate = _coefficient * (1 + z) ** 2.7 / (1 + ((1 + z) / 2.9) ** 5.6)

    return rate if z.ndim > 0 else rate.item()  # Return scalar if input was scalar.


class TypeIIPSNe(ExtragalacticTransient):
    """Type IIP core-collapse SNe: `TypeIIPSED` (two-exponential + radioactive-tail lightcurve x cooling blackbody)."""

    DEFAULT_MODEL = TypeIIPSED
    DEFAULT_DURATION = 100 * u.day
    DEFAULT_Z_LIM = 0.8

    def event_rate(self, z: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
        """Volumetric event rate: `_core_collapse_rate(z)` times the Type IIP fraction (see module docstring)."""
        return _TYPE_IIP_FRACTION * _core_collapse_rate(z)


class TypeIIPExcessSNe(ExtragalacticTransient):
    """Early-interacting (IXF/GGI-like) Type IIP core-collapse SNe: `TypeIIPExcessSED`."""

    DEFAULT_MODEL = TypeIIPExcessSED
    DEFAULT_DURATION = 100 * u.day
    DEFAULT_Z_LIM = 2

    def event_rate(self, z: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
        """Volumetric event rate: `_core_collapse_rate(z)` times the Type IIP-excess fraction (see module docstring)."""
        return _TYPE_IIP_EXCESS_FRACTION * _core_collapse_rate(z)
