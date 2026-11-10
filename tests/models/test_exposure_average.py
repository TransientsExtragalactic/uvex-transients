"""
Tests for averaging a model's flux over an exposure.

`exposure_average_nodes` supplies the quadrature, `SpectralModel.as_source_spectrum` (``exposure=``) applies it, and
`SpectralModel.measure_photometry` / `simulate_photometry` (``average_exposure=``) measure with it. The reference for
every number is an adaptive integral of the same model.
"""

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from m4opt.missions._uvex import uvex
from scipy.integrate import quad

from uvex_transients.models._utils import N_EXPOSURE_NODES, exposure_average_nodes
from uvex_transients.phase_space.transient import _FREDPowerLawSED

EXPOSURE = 900.0
"""float: The UVEX dwell, in seconds."""

WAVE = np.array([1500.0])
"""numpy.ndarray: Wavelength, in Angstrom, at which spectra are compared."""

REDSHIFT = 0.02


def _parameters(rise: float, decay: float) -> dict:
    """Return the parameters of a FRED light curve with a power-law spectrum, with timescales in seconds."""
    return {
        "amplitude": 1e43 * u.erg / u.s,
        "rise": rise * u.s,
        "decay": decay * u.s,
        "spectral_index": -2.0,
        "reference_frequency": 1e15 * u.Hz,
        "frequency_min": 1e14 * u.Hz,
        "frequency_max": 1e17 * u.Hz,
    }


def _batched(parameters: dict) -> dict:
    """Give every parameter the ``(1, 1)`` shape (epoch, wavelength) that `measure_photometry` hands the model."""
    return {
        name: np.asarray(getattr(value, "value", value))[None, None] * getattr(value, "unit", 1.0)
        for name, value in parameters.items()
    }


def _flux_at(t: float, parameters: dict) -> float:
    """Return the instantaneous photon flux density at `WAVE`, `t` seconds after the explosion."""
    spectrum = _FREDPowerLawSED.as_source_spectrum(np.array([t]) * u.s, redshift=REDSHIFT, **parameters)
    return float(np.asarray(spectrum(WAVE).value).ravel()[0])


def _averaged_flux(t: float, parameters: dict, **kwargs) -> float:
    """Return the photon flux density at `WAVE` averaged over an exposure starting `t` seconds after the explosion."""
    spectrum = _FREDPowerLawSED.as_source_spectrum(
        np.array([[t]]) * u.s, redshift=REDSHIFT, exposure=EXPOSURE * u.s, **_batched(parameters), **kwargs
    )
    return float(np.asarray(spectrum(WAVE).value).ravel()[0])


def _exact_average(t: float, parameters: dict) -> float:
    """Return the mean over the exposure by adaptive integration, with no source before the explosion."""
    start, stop = max(t, 0.0), t + EXPOSURE
    breakpoints = [start + x for x in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0) if start + x < stop]
    integral, _ = quad(
        lambda time: _flux_at(time, parameters),
        start,
        stop,
        points=breakpoints or None,
        limit=500,
        epsabs=0.0,
        epsrel=1e-9,
    )
    return integral / EXPOSURE


# --------------------------------------------------------------------------- #
# The quadrature                                                              #
# --------------------------------------------------------------------------- #
class TestExposureAverageNodes:
    START = np.array([0.0, 50.0, -300.0, -900.0, -1500.0])

    def test_shape(self):
        nodes, weights = exposure_average_nodes(self.START, EXPOSURE)
        assert nodes.shape == weights.shape == (N_EXPOSURE_NODES, self.START.size)
        assert N_EXPOSURE_NODES == 28

    def test_weights_sum_to_the_fraction_of_the_exposure_after_the_explosion(self):
        _, weights = exposure_average_nodes(self.START, EXPOSURE)
        np.testing.assert_allclose(weights.sum(axis=0), [1.0, 1.0, 2.0 / 3.0, 0.0, 0.0], atol=1e-14)

    def test_nodes_lie_in_the_exposure_after_the_explosion(self):
        nodes, weights = exposure_average_nodes(self.START, EXPOSURE)
        lower = np.maximum(self.START, 0.0)
        upper = np.maximum(self.START + EXPOSURE, 0.0)
        assert np.all(nodes >= lower - 1e-9)
        assert np.all(nodes <= upper + 1e-9)

        # No exposure after the explosion means nothing to integrate, and every node sits at the explosion
        assert np.all(nodes[:, -2:] == 0.0)

    def test_integrates_a_cubic_exactly(self):
        nodes, weights = exposure_average_nodes(self.START, EXPOSURE)
        lower = np.maximum(self.START, 0.0)
        upper = np.maximum(self.START + EXPOSURE, 0.0)
        exact = (upper**4 - lower**4) / 4.0 / EXPOSURE
        np.testing.assert_allclose((weights * nodes**3).sum(axis=0), exact, rtol=1e-12, atol=1e-12)

    def test_resolves_the_first_instants_after_the_start(self):
        nodes, _ = exposure_average_nodes(np.array([0.0]), EXPOSURE)
        assert nodes.min() < 0.1

    def test_broadcasts_a_scalar_and_an_array_exposure(self):
        nodes, weights = exposure_average_nodes(self.START, np.full(self.START.shape, EXPOSURE))
        nodes_scalar, weights_scalar = exposure_average_nodes(self.START, EXPOSURE)
        np.testing.assert_array_equal(nodes, nodes_scalar)
        np.testing.assert_array_equal(weights, weights_scalar)

    def test_batch_axes_are_kept(self):
        nodes, weights = exposure_average_nodes(self.START[:, None], EXPOSURE)
        assert nodes.shape == weights.shape == (N_EXPOSURE_NODES, self.START.size, 1)


# --------------------------------------------------------------------------- #
# The averaged spectrum                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rise, decay", [(3.0, 30.0), (0.5, 5.0), (100.0, 1000.0)])
@pytest.mark.parametrize("start", [0.0, 5.0, 100.0, -300.0])
def test_average_matches_an_adaptive_integral(rise, decay, start):
    """Pulses from a few seconds to a thousand, starting before, at and after the explosion, to 1e-3."""
    parameters = _parameters(rise, decay)
    np.testing.assert_allclose(_averaged_flux(start, parameters), _exact_average(start, parameters), rtol=1e-3)


def test_straddling_exposure_is_diluted_by_the_part_before_the_explosion():
    """An exposure that starts before the explosion collects the same photons as one that starts at it, per exposure."""
    parameters = _parameters(3.0, 30.0)

    # The pulse is over within the 600 s after the explosion, so this exposure holds all of its photons, as one that
    # starts at the explosion does
    np.testing.assert_allclose(_averaged_flux(-300.0, parameters), _averaged_flux(0.0, parameters), rtol=1e-3)

    # One that ends before the pulse has started sees none
    assert _averaged_flux(-1000.0, parameters) == 0.0

    # A source that is constant after the explosion is diluted by exactly the fraction of the exposure it is on for
    constant = _parameters(1e-3, 1e12)
    constant["amplitude"] = 1e43 * u.erg / u.s
    on = _averaged_flux(0.0, constant)
    np.testing.assert_allclose(_averaged_flux(-300.0, constant), on * 600.0 / EXPOSURE, rtol=1e-3)


def test_a_slowly_varying_source_is_the_same_averaged_or_not():
    parameters = _parameters(1e7, 1e8)
    np.testing.assert_allclose(_averaged_flux(1e7, parameters), _flux_at(1e7 + EXPOSURE / 2, parameters), rtol=1e-3)


def test_a_short_pulse_is_much_fainter_averaged_than_at_its_peak():
    parameters = _parameters(3.0, 30.0)
    assert _averaged_flux(5.0, parameters) < 0.1 * _flux_at(5.0, parameters)


def test_dust_attenuation_multiplies_the_average():
    """The attenuation does not change in time, so it factors out of the mean."""
    parameters = _parameters(3.0, 30.0)
    with_dust = _averaged_flux(0.0, parameters, ebv=0.2)
    without_dust = _averaged_flux(0.0, parameters)

    instant = _FREDPowerLawSED.as_source_spectrum(np.array([20.0]) * u.s, redshift=REDSHIFT, ebv=0.2, **parameters)
    instant_without = _FREDPowerLawSED.as_source_spectrum(np.array([20.0]) * u.s, redshift=REDSHIFT, **parameters)
    ratio = float(np.asarray(instant(WAVE).value).ravel()[0] / np.asarray(instant_without(WAVE).value).ravel()[0])

    assert ratio < 1.0
    np.testing.assert_allclose(with_dust / without_dust, ratio, rtol=1e-9)


def test_default_is_the_flux_at_the_start():
    """Without `exposure` the spectrum is the one at the start of the exposure, as it always was."""
    parameters = _parameters(3.0, 30.0)
    spectrum = _FREDPowerLawSED.as_source_spectrum(np.array([[20.0]]) * u.s, redshift=REDSHIFT, **_batched(parameters))
    np.testing.assert_array_equal(np.asarray(spectrum(WAVE).value).ravel(), [_flux_at(20.0, parameters)])


# --------------------------------------------------------------------------- #
# Photometry                                                                  #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def measurement():
    """Return a function that measures the short pulse at given exposure starts, with or without averaging."""
    coord = SkyCoord(150 * u.deg, 20 * u.deg)
    parameters = _parameters(3.0, 30.0)
    model = _FREDPowerLawSED()

    def measure(starts, **kwargs):
        starts = np.asarray(starts, dtype=float) * u.s
        return model.measure_photometry(
            starts,
            EXPOSURE * u.s,
            uvex.detector,
            coord,
            bands=["NUV"],
            redshift=REDSHIFT,
            noise_seed=1,
            noise_observation_keys=np.arange(starts.size, dtype=np.uint64),
            **parameters,
            **kwargs,
        )

    return measure


def test_averaging_lowers_the_expected_snr_of_a_short_pulse(measurement):
    """At the peak of a 30 s pulse the flux at the start of a 900 s exposure is far above its mean."""
    _, _, _, snr_start = measurement([5.0])
    _, _, _, snr_average = measurement([5.0], average_exposure=True)
    assert snr_average[0, 0] < 0.5 * snr_start[0, 0]


def test_default_measurement_is_unchanged_by_the_option_existing(measurement):
    """`average_exposure=False` is the default, and gives identical numbers."""
    for default, explicit in zip(measurement([5.0, 100.0]), measurement([5.0, 100.0], average_exposure=False)):
        np.testing.assert_array_equal(default, explicit)


def test_exposures_that_end_after_the_explosion_have_a_source_when_averaging(measurement):
    """An exposure is background only if it ends before the explosion, not if it merely starts before it."""
    starts = [-1000.0, -895.0, -300.0, 0.0]
    _, _, _, snr_start = measurement(starts)
    _, _, _, snr_average = measurement(starts, average_exposure=True)

    # Without averaging every exposure that starts before the explosion is blank
    blank = snr_start[0, 0]
    np.testing.assert_allclose(snr_start[0, :2], blank, rtol=1e-6)

    # With it only the one that ends before the explosion is, and the straddling ones see the source
    np.testing.assert_allclose(snr_average[0, 0], blank, rtol=1e-6)
    assert snr_average[0, 1] > 1.001 * blank
    assert snr_average[0, 2] > snr_average[0, 1]


def test_simulate_photometry_marks_straddling_exposures_as_in_the_model():
    coord = SkyCoord(150 * u.deg, 20 * u.deg)
    kwargs = {"bands": ["NUV"], "redshift": REDSHIFT, "noise_seed": 1, **_parameters(3.0, 30.0)}
    starts = np.array([-1000.0, -300.0, 0.0, 100.0]) * u.s
    keys = np.arange(starts.size, dtype=np.uint64)

    table = _FREDPowerLawSED().simulate_photometry(
        starts, EXPOSURE * u.s, uvex.detector, coord, average_exposure=True, noise_observation_keys=keys, **kwargs
    )
    assert list(table["in_model"]) == [False, True, True, True]

    default = _FREDPowerLawSED().simulate_photometry(
        starts, EXPOSURE * u.s, uvex.detector, coord, noise_observation_keys=keys, **kwargs
    )
    assert list(default["in_model"]) == [False, False, True, True]
