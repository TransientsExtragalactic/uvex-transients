"""
Tests for :mod:`uvex_transients.models.lightcurves`.

Each concrete :class:`~uvex_transients.models.core.base.Lightcurve` gets a two-line
test class inheriting the generic checks from
:class:`~uvex_transients.models.tests._contracts.LightcurveContract`. See that module's
docstring for what is actually being checked.
"""

import numpy as np
import pytest

from uvex_transients.models.core.base import Lightcurve
from uvex_transients.models.lightcurves import (
    BazinLightcurve,
    BrokenPowerLawLightcurve,
    DelayedExponentialLightcurve,
    FREDLightcurve,
    GaussianPulseLightcurve,
    GaussianRiseBrokenPowerLawLightcurve,
    GaussianRisePowerLawLightcurve,
    GREDLightcurve,
    InverseGammaBrokenPowerLawLightcurve,
    KhatamiSBOLightcurve,
    LogNormalPulseLightcurve,
    PlateauPowerLawLightcurve,
    PowerLawLightcurve,
    SBOPhenomenologicalLightcurve,
    SmoothBrokenPowerLawLightcurve,
    TopHatLightcurve,
    TwoComponentBazinLightcurve,
    VillarLightcurve,
)

from ._contracts import LightcurveContract, assert_full_coverage


class TestTopHatLightcurve(LightcurveContract):
    model_class = TopHatLightcurve


class TestGaussianPulseLightcurve(LightcurveContract):
    model_class = GaussianPulseLightcurve


class TestFREDLightcurve(LightcurveContract):
    model_class = FREDLightcurve


class TestGREDLightcurve(LightcurveContract):
    model_class = GREDLightcurve


class TestGaussianRisePowerLawLightcurve(LightcurveContract):
    model_class = GaussianRisePowerLawLightcurve


class TestGaussianRiseBrokenPowerLawLightcurve(LightcurveContract):
    model_class = GaussianRiseBrokenPowerLawLightcurve


class TestBazinLightcurve(LightcurveContract):
    model_class = BazinLightcurve


class TestPowerLawLightcurve(LightcurveContract):
    model_class = PowerLawLightcurve


class TestBrokenPowerLawLightcurve(LightcurveContract):
    model_class = BrokenPowerLawLightcurve


class TestSmoothBrokenPowerLawLightcurve(LightcurveContract):
    model_class = SmoothBrokenPowerLawLightcurve


class TestDelayedExponentialLightcurve(LightcurveContract):
    model_class = DelayedExponentialLightcurve


class TestLogNormalPulseLightcurve(LightcurveContract):
    model_class = LogNormalPulseLightcurve


class TestPlateauPowerLawLightcurve(LightcurveContract):
    model_class = PlateauPowerLawLightcurve


class TestVillarLightcurve(LightcurveContract):
    model_class = VillarLightcurve


class TestTwoComponentBazinLightcurve(LightcurveContract):
    model_class = TwoComponentBazinLightcurve


class TestInverseGammaBrokenPowerLawLightcurve(LightcurveContract):
    model_class = InverseGammaBrokenPowerLawLightcurve

    @pytest.mark.parametrize(
        "p, q, x_break, smoothness",
        [(2.0, 1.0, 5.0, 2.0), (1.0, 3.0, 0.5, 4.0), (2.0, 2.0, 1.0, 1.0), (4.0, 0.5, 30.0, 8.0)],
    )
    def test_peak_equals_amplitude(self, p, q, x_break, smoothness):
        """The normalization makes the maximum of the light curve exactly `amplitude`."""
        t = np.geomspace(1e-3, 1e5, 200001)
        lum = InverseGammaBrokenPowerLawLightcurve.eval_cgs(
            t, amplitude=3.0, t_scale=1.0, p=p, q=q, x_break=x_break, smoothness=smoothness
        )
        assert lum.max() <= 3.0 * (1 + 1e-12)
        np.testing.assert_allclose(lum.max(), 3.0, rtol=1e-6)

    def test_zero_before_explosion(self):
        lum = InverseGammaBrokenPowerLawLightcurve.eval_cgs(
            np.array([-1.0, 0.0]), amplitude=1.0, t_scale=1.0, p=2.0, q=1.0, x_break=5.0, smoothness=2.0
        )
        np.testing.assert_array_equal(lum, 0.0)


class TestSBOPhenomenologicalLightcurve(LightcurveContract):
    model_class = SBOPhenomenologicalLightcurve


class TestKhatamiSBOLightcurve(LightcurveContract):
    model_class = KhatamiSBOLightcurve


def test_all_lightcurves_covered():
    """Fail loudly if a new `Lightcurve` subclass is added without a `Test*` class above."""
    tested = {
        cls.model_class
        for name, cls in globals().items()
        if name.startswith("Test") and issubclass(cls, LightcurveContract)
    }
    assert_full_coverage(Lightcurve, tested)
