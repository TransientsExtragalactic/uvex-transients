"""Tests for the luminosity function of a transient class in a band."""

import numpy as np
import pytest
import synphot
from astropy import units as u

from uvex_transients.models.core.priors import PowerLawPrior
from uvex_transients.phase_space.transient import _CanonicalTransient
from uvex_transients.transients.base import TransientBase
from uvex_transients.transients.kilonovae import Kilonova
from uvex_transients.utils.phot_utils import equivalent_bandwidth, luminosity_to_absolute_magnitude

L_NU = 1e29 * u.erg / u.s / u.Hz
GRID = {"t_min": 0.05 * u.day, "t_max": 200 * u.day, "n_time": 150}


@pytest.fixture(scope="module")
def bandpass():
    return synphot.SpectralElement(synphot.Gaussian1D, mean=1500 * u.AA, stddev=100 * u.AA)


@pytest.fixture(scope="module")
def point(bandpass):
    """A canonical transient with every parameter pinned, so every draw has the same peak."""
    transient = _CanonicalTransient(spectral_index=0.0)
    transient.set_point(L_NU, 10 * u.day, 1.0)
    return transient


@pytest.fixture(scope="module")
def log_uniform(bandpass):
    """A canonical transient whose peak luminosity is log-uniform over two decades, 1e42 to 1e44 erg/s bolometric."""
    transient = _CanonicalTransient(spectral_index=0.0)
    transient.set_point(L_NU, 10 * u.day, 1.0)
    amplitude = transient.sed["amplitude"]
    amplitude.unfix()
    # The amplitude is sampled as A / 1e43 erg/s, and a power law of index 1 is uniform in its log
    amplitude.set_prior(PowerLawPrior(alpha=1.0, lower=0.1, upper=10.0))
    return transient


def test_every_event_of_a_pinned_transient_lands_in_the_bin_of_its_peak(point, bandpass):
    peak = (L_NU * equivalent_bandwidth(bandpass)).to_value(u.erg / u.s)
    edges = peak * np.array([0.9, 0.99, 1.01, 1.1]) * u.erg / u.s
    table = point.luminosity_function(bandpass, 50, bins=edges, rng=0, **GRID)
    np.testing.assert_array_equal(table["count"], [0, 50, 0])
    assert table["pdf"][1].value == pytest.approx(1 / (0.02 * peak), rel=3e-3)
    assert table.meta["n_valid"] == 50 and table.meta["quantity"] == "luminosity"
    assert table.meta["class"] == "_CanonicalTransient"


def test_a_log_uniform_population_is_flat_in_log_luminosity(log_uniform, bandpass):
    table = log_uniform.luminosity_function(bandpass, 6000, bins=8, rng=1, **GRID)
    assert table["lo"].unit == u.erg / u.s and table["pdf"].unit == (u.erg / u.s) ** -1
    # Equal-width bins in log L hold equal counts, so dN/dL falls as 1/L
    centre = np.sqrt(table["lo"] * table["hi"])
    assert np.all(table["count"] > 0.8 * 6000 / 8) and np.all(table["count"] < 1.2 * 6000 / 8)
    np.testing.assert_allclose((table["pdf"] * centre).value, np.mean((table["pdf"] * centre).value), rtol=0.25)
    # The density is normalized
    assert np.sum(table["pdf"] * (table["hi"] - table["lo"])).to_value(u.dimensionless_unscaled) == pytest.approx(1.0)
    # and each bin's interval holds its expected count of n / 8 at one sigma, or close to it
    outside = (table["count_lo"] > 6000 / 8) | (table["count_hi"] < 6000 / 8)
    assert outside.sum() <= 3


def test_the_magnitude_function_is_the_luminosity_function_reversed(log_uniform, bandpass):
    luminosity = log_uniform.luminosity_function(bandpass, 500, bins=6, rng=2, **GRID)
    magnitude = log_uniform.luminosity_function(bandpass, 500, quantity="magnitude", bins=6, rng=2, **GRID)
    assert magnitude["lo"].unit == u.ABmag and magnitude["pdf"].unit == u.mag**-1
    assert np.all(np.diff(magnitude["lo"].value) > 0)
    assert magnitude.meta["quantity"] == "magnitude"

    # The same events binned in two variables: the extremes of the sample are the extremes of the other
    bright = luminosity_to_absolute_magnitude(luminosity["hi"][-1], bandpass)
    faint = luminosity_to_absolute_magnitude(luminosity["lo"][0], bandpass)
    assert magnitude["lo"][0].value == pytest.approx(bright.value, abs=1e-6)
    assert magnitude["hi"][-1].value == pytest.approx(faint.value, abs=1e-6)
    assert np.sum(magnitude["pdf"] * (magnitude["hi"] - magnitude["lo"]).value).value == pytest.approx(1.0)


def test_explicit_magnitude_edges_are_used_as_given(log_uniform, bandpass):
    edges = np.arange(-30.0, 10.0, 5.0) * u.ABmag
    table = log_uniform.luminosity_function(bandpass, 200, quantity="magnitude", bins=edges, rng=3, **GRID)
    np.testing.assert_allclose(table["lo"].value, edges.value[:-1])
    np.testing.assert_allclose(table["hi"].value, edges.value[1:])
    plain = log_uniform.luminosity_function(bandpass, 200, quantity="magnitude", bins=edges.value, rng=3, **GRID)
    np.testing.assert_array_equal(plain["count"], table["count"])


def test_the_time_grid_defaults_come_from_the_class(bandpass):
    kilonova = Kilonova()
    default = kilonova.luminosity_function(bandpass, 30, bins=5, rng=4)
    explicit = kilonova.luminosity_function(
        bandpass,
        30,
        bins=5,
        rng=4,
        t_min=0.01 * u.day,
        t_max=5 * kilonova.DEFAULT_DURATION,
        n_time=300,
    )
    np.testing.assert_array_equal(default["count"], explicit["count"])


def test_events_with_no_measurable_peak_are_dropped_and_counted(point, bandpass, monkeypatch):
    peaks = np.array([1e40, np.nan, 0.0, 1e41]) * u.erg / u.s
    monkeypatch.setattr(type(point.sed), "sample_peak_luminosities", lambda self, *a, **k: peaks)
    table = point.luminosity_function(bandpass, 4, bins=2, **GRID)
    assert table.meta["n"] == 4 and table.meta["n_valid"] == 2 and table["count"].sum() == 2

    monkeypatch.setattr(type(point.sed), "sample_peak_luminosities", lambda self, *a, **k: peaks * np.nan)
    with pytest.raises(ValueError, match="measurable"):
        point.luminosity_function(bandpass, 4, bins=2, **GRID)


def test_bad_arguments_are_rejected(point, bandpass):
    with pytest.raises(ValueError, match="quantity"):
        point.luminosity_function(bandpass, 5, quantity="flux", **GRID)
    with pytest.raises(ValueError, match="bins"):
        point.luminosity_function(bandpass, 5, bins=0, **GRID)
    with pytest.raises(ValueError, match="edges"):
        point.luminosity_function(bandpass, 5, bins=[3.0, 2.0], **GRID)


def test_a_seed_reproduces_the_function(log_uniform, bandpass):
    first = log_uniform.luminosity_function(bandpass, 100, bins=5, rng=11, **GRID)
    second = log_uniform.luminosity_function(bandpass, 100, bins=5, rng=11, **GRID)
    np.testing.assert_array_equal(first["count"], second["count"])
    np.testing.assert_array_equal(first["lo"].value, second["lo"].value)


def test_every_registered_class_has_a_luminosity_function(bandpass):
    """A class added later without a measurable peak in a UV band fails here, not in a user's run."""
    # The registry fills as modules are imported
    import uvex_transients.phase_space.overlay as overlay  # noqa: F401  (imports the shipped transient modules)

    classes = {name: cls for name, cls in TransientBase.registry().items() if not name.startswith("_")}
    assert classes
    for name, cls in classes.items():
        table = cls().luminosity_function(bandpass, 20, bins=4, rng=0)
        assert table["count"].sum() > 0, name
        assert table.meta["class"] == name
