"""Tests for placing the shipped transient classes on a map."""

import numpy as np
import pytest
import synphot
from astropy import units as u

from uvex_transients.phase_space.overlay import measure_class, shipped_classes, split_measurements, stack_measurements
from uvex_transients.transients.kilonovae import Kilonova
from uvex_transients.transients.TDEs import TidalDisruptionEvent


@pytest.fixture(scope="module")
def bandpass():
    return synphot.SpectralElement(synphot.Gaussian1D, mean=1500 * u.AA, stddev=100 * u.AA)


def test_the_shipped_classes_are_the_public_extragalactic_ones():
    classes = shipped_classes()
    assert {"Kilonova", "TidalDisruptionEvent", "LuminousFastBlueOpticalTransient", "TypeIaSNe"} <= set(classes)
    assert not any(name.startswith("_") for name in classes)
    assert "_CanonicalTransient" not in classes


def test_the_bespoke_shock_cooling_class_is_left_off_the_maps_but_the_type_iib_population_is_on():
    # ShockCoolingIIb's SED is zero outside its validity window, which sets its half-decline time.
    classes = shipped_classes()
    assert "ShockCoolingIIb" not in classes
    assert "TypeIIbSNe" in classes


def test_a_class_is_measured_in_the_band_of_the_map(bandpass):
    table = measure_class(TidalDisruptionEvent, bandpass, n_draws=60, rng=0)
    assert len(table) == 60
    assert table.meta["class"] == "TidalDisruptionEvent"
    assert set(table.colnames) == {"log_luminosity", "log_half_time", "measured"}
    good = np.asarray(table["measured"])
    assert good.mean() > 0.9
    assert np.all(np.isfinite(table["log_luminosity"][good])) and np.all(np.isfinite(table["log_half_time"][good]))
    # A tidal disruption in the UV is a luminous, slow event.
    assert 41 < np.median(table["log_luminosity"][good]) < 45
    assert 0 < np.median(table["log_half_time"][good]) < 3


def test_the_parameters_of_a_class_are_drawn_independently(bandpass):
    # Seeding every parameter alike would make the draws one-dimensional, and the measured L and t_1/2 collinear.
    table = measure_class(Kilonova, bandpass, n_draws=400, rng=0)
    good = np.asarray(table["measured"])
    assert abs(np.corrcoef(table["log_luminosity"][good], table["log_half_time"][good])[0, 1]) < 0.9


def test_the_same_seed_repeats_and_a_wider_grid_leaves_the_peak_unchanged(bandpass):
    first = measure_class(TidalDisruptionEvent, bandpass, n_draws=20, rng=3)
    again = measure_class(TidalDisruptionEvent, bandpass, n_draws=20, rng=3)
    longer = measure_class(TidalDisruptionEvent, bandpass, n_draws=20, rng=3, t_max=2000 * u.day)
    np.testing.assert_array_equal(first["log_luminosity"], again["log_luminosity"])
    np.testing.assert_allclose(first["log_luminosity"], longer["log_luminosity"], atol=0.02)


def test_stacking_and_splitting_measurements_round_trips(bandpass, tmp_path):
    tables = [
        measure_class(TidalDisruptionEvent, bandpass, n_draws=30, rng=0),
        measure_class(Kilonova, bandpass, n_draws=20, rng=0),
    ]
    stacked = stack_measurements(tables)
    assert len(stacked) == 50 and stacked.colnames == ["class", "log_luminosity", "log_half_time", "measured"]

    # Through a file, as a map's overlay.ecsv travels
    path = tmp_path / "overlay.ecsv"
    stacked.write(path, format="ascii.ecsv")
    from astropy.table import Table

    restored = split_measurements(Table.read(path, format="ascii.ecsv"))
    assert [table.meta["class"] for table in restored] == ["TidalDisruptionEvent", "Kilonova"]
    for before, after in zip(tables, restored):
        assert after.colnames == ["log_luminosity", "log_half_time", "measured"]
        np.testing.assert_array_equal(np.asarray(before["measured"]), np.asarray(after["measured"]))
        good = np.asarray(before["measured"])
        np.testing.assert_allclose(
            np.asarray(before["log_luminosity"])[good], np.asarray(after["log_luminosity"])[good]
        )
