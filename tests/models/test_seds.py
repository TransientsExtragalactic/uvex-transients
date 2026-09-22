"""
Tests for the composite SED models in :mod:`uvex_transients.models.supernovae`,
:mod:`uvex_transients.models.tdes`, and :mod:`uvex_transients.models.lfbots`.

Each concrete :class:`~uvex_transients.models.core.base.SpectralModel` gets a two-line
test class inheriting the generic checks from
:class:`~uvex_transients.models.tests._contracts.SpectralModelContract`. This covers
both plain `SpectralModel` subclasses (e.g. `VillarCoolingBlackbodySED`) and
`ComposedSpectralModel` subclasses (e.g. `VanVelzenTDESED`) -- both expose
exactly the same public interface. See `_contracts`'s docstring for what is
actually being checked.
"""

import numpy as np
from astropy import units as u

from uvex_transients.models.core.base import ComposedSpectralModel, SpectralModel
from uvex_transients.models.kilonovae import KilonovaCoolingBlackbodySED
from uvex_transients.models.lfbots import LFBOTCoolingBlackbodySED
from uvex_transients.models.supernovae import (
    ArnettMagnetarSpindownSED,
    MoragShockCoolingBlackbodySED,
    MoragShockCoolingSED,
    TypeIbSED,
    TypeIcSED,
    TypeIIbSED,
    TypeIIPExcessSED,
    TypeIIPSED,
    VillarCoolingBlackbodySED,
)
from uvex_transients.models.tdes import AlushStoneTDESED, VanVelzenTDESED

from ._contracts import SpectralModelContract, assert_full_coverage


class TestArnettMagnetarSpindownSED(SpectralModelContract):
    model_class = ArnettMagnetarSpindownSED


class TestVillarCoolingBlackbodySED(SpectralModelContract):
    model_class = VillarCoolingBlackbodySED


class TestMoragShockCoolingSED(SpectralModelContract):
    """Narrower `t_grid` than the default: Morag+24's shock-cooling fit is only valid for a
    fairly short early-time window (this model masks parameter draws outside it to `nan`), unlike
    the other phenomenological SEDs here which stay finite across the full default grid."""

    model_class = MoragShockCoolingSED
    t_grid: u.Quantity = np.geomspace(3e-2, 5, 24) * u.day


class TestMoragShockCoolingBlackbodySED(SpectralModelContract):
    model_class = MoragShockCoolingBlackbodySED
    t_grid: u.Quantity = np.geomspace(3e-2, 5, 24) * u.day


class TestTypeIIPSED(SpectralModelContract):
    model_class = TypeIIPSED


class TestTypeIIPExcessSED(SpectralModelContract):
    model_class = TypeIIPExcessSED


class TestTypeIIbSED(SpectralModelContract):
    model_class = TypeIIbSED


class TestTypeIbSED(SpectralModelContract):
    model_class = TypeIbSED


class TestTypeIcSED(SpectralModelContract):
    model_class = TypeIcSED


class TestVanVelzenTDESED(SpectralModelContract):
    model_class = VanVelzenTDESED


class TestAlushStoneTDESED(SpectralModelContract):
    model_class = AlushStoneTDESED


class TestLFBOTCoolingBlackbodySED(SpectralModelContract):
    model_class = LFBOTCoolingBlackbodySED


class TestKilonovaCoolingBlackbodySED(SpectralModelContract):
    model_class = KilonovaCoolingBlackbodySED


def test_all_seds_covered():
    """Fail loudly if a new `SpectralModel` subclass is added without a `Test*` class above.

    `ComposedSpectralModel` itself is excluded: it is an extension point
    (`_LIGHTCURVE_CLASS`/`_SPECTRUM_CLASS` unset), not a real model.
    """
    tested = {
        cls.model_class
        for name, cls in globals().items()
        if name.startswith("Test") and issubclass(cls, SpectralModelContract)
    }
    assert_full_coverage(SpectralModel, tested, exclude={ComposedSpectralModel})
