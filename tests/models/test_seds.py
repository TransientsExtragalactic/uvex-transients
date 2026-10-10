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
import pytest
from astropy import units as u

from uvex_transients.models.arnett import ArnettDecaySED
from uvex_transients.models.core.base import ComposedSpectralModel, SpectralModel
from uvex_transients.models.core.priors import PowerLawPrior
from uvex_transients.models.kilonovae import KilonovaCoolingBlackbodySED
from uvex_transients.models.lfbots import LFBOTCoolingBlackbodySED
from uvex_transients.models.supernovae import (
    ArnettMagnetarSpindownSED,
    MoragShockCoolingBlackbodySED,
    MoragShockCoolingSED,
    TypeIaSED,
    TypeIbSED,
    TypeIcBLSED,
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


class TestArnettDecaySED(SpectralModelContract):
    model_class = ArnettDecaySED


class TestTypeIaSED(SpectralModelContract):
    model_class = TypeIaSED


class TestVillarCoolingBlackbodySED(SpectralModelContract):
    model_class = VillarCoolingBlackbodySED


class _UnmaskedMoragContract(SpectralModelContract):
    """Run the contract on the raw Morag+24 formulas, with `_INVALID_FILL` off.

    The contract asserts finite output everywhere, but with the default fill the model is zero (or ``nan``)
    outside its regime of validity, which for Type IIb parameters is only a day or two. The masking itself is
    covered by `test_morag_validity`. The `t_grid` is also narrower than the default, because the raw formulas
    are only well behaved over a fairly short early-time window.
    """

    t_grid: u.Quantity = np.geomspace(3e-2, 5, 24) * u.day

    @pytest.fixture(autouse=True)
    def _disable_masking(self, monkeypatch):
        monkeypatch.setattr(self.model_class, "_INVALID_FILL", None)


class TestMoragShockCoolingSED(_UnmaskedMoragContract):
    model_class = MoragShockCoolingSED


class TestMoragShockCoolingBlackbodySED(_UnmaskedMoragContract):
    model_class = MoragShockCoolingBlackbodySED


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


class TestTypeIcBLSED(SpectralModelContract):
    model_class = TypeIcBLSED


class TestVanVelzenTDESED(SpectralModelContract):
    model_class = VanVelzenTDESED


class TestAlushStoneTDESED(SpectralModelContract):
    model_class = AlushStoneTDESED


@pytest.mark.parametrize("model_class", [VanVelzenTDESED, AlushStoneTDESED])
def test_tde_amplitude_prior_is_the_yao_luminosity_function(model_class):
    """The TDE peak-luminosity prior is dN/dlogL ~ L^-1.41 above 1e43 erg/s, i.e. p(L) ~ L^-2.41."""
    prior = model_class()["amplitude"].prior
    assert prior == PowerLawPrior(alpha=2.41, lower=1.0e43, upper=np.inf)

    L = model_class().sample_parameters(size=200_000, rng=0)["amplitude"].to_value(u.erg / u.s)
    assert L.min() >= 1.0e43
    assert np.median(L) == pytest.approx(1.0e43 * 2.0 ** (1.0 / 1.41), rel=0.02)


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
