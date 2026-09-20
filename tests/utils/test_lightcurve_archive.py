"""Tests for `uvex_transients.utils.lightcurve_archive.LightcurveArchive`."""

import pytest
from astropy import units as u
from astropy.table import QTable

from uvex_transients.utils.lightcurve_archive import LightcurveArchive


@pytest.fixture
def archive_path(tmp_path):
    """Build a small synthetic archive with two transient types, mirroring the real schema."""
    path = tmp_path / "lightcurves.h5"

    lbol = QTable()
    lbol["time"] = [1.0, 2.0, 3.0] * u.day
    lbol["L_bol"] = [1e42, 2e42, 3e42] * (u.erg / u.s)
    lbol.meta["reference"] = "https://ui.adsabs.harvard.edu/abs/2016MNRAS.457..328L/abstract"
    lbol.write(path, path="supernovae/1999xx_fakeref/L_bol", format="hdf5", append=True, serialize_meta=True)

    tphot = QTable()
    tphot["time"] = [1.0, 2.0] * u.day
    tphot["T_phot"] = [10000.0, 9000.0] * u.K
    tphot.meta["reference"] = "https://ui.adsabs.harvard.edu/abs/2016MNRAS.457..328L/abstract"
    tphot.write(path, path="supernovae/1999xx_fakeref/T_phot", format="hdf5", append=True, serialize_meta=True)

    other = QTable()
    other["time"] = [0.5] * u.day
    other["L_bol"] = [5e41] * (u.erg / u.s)
    other.write(path, path="kilonovae/gw000000_fakeref/L_bol", format="hdf5", append=True, serialize_meta=True)

    return path


def test_types_lists_top_level_groups(archive_path):
    """`types` returns every transient-type group, sorted."""
    archive = LightcurveArchive(archive_path)
    assert archive.types() == ["kilonovae", "supernovae"]


def test_events_lists_transients_under_a_type(archive_path):
    """`events` returns every ``<designation>_<citekey>`` transient under one type."""
    archive = LightcurveArchive(archive_path)
    assert archive.events("supernovae") == ["1999xx_fakeref"]
    assert archive.events("kilonovae") == ["gw000000_fakeref"]


def test_fields_excludes_hdf5_metadata_sidecars(archive_path):
    """`fields` only reports real observables, not astropy's ``__table_column_meta__`` sidecars."""
    archive = LightcurveArchive(archive_path)
    assert archive.fields("supernovae", "1999xx_fakeref") == ["L_bol", "T_phot"]


def test_table_round_trips_values_units_and_reference(archive_path):
    """`table` returns unit-aware columns and the stored reference, matching what was written."""
    archive = LightcurveArchive(archive_path)
    table = archive.table("supernovae", "1999xx_fakeref", "L_bol")

    assert table["time"].unit == u.day
    assert table["L_bol"].unit == u.erg / u.s
    assert table["time"].to_value(u.day) == pytest.approx([1.0, 2.0, 3.0])
    assert table["L_bol"].to_value(u.erg / u.s) == pytest.approx([1e42, 2e42, 3e42])
    assert table.meta["reference"] == "https://ui.adsabs.harvard.edu/abs/2016MNRAS.457..328L/abstract"


def test_table_missing_path_raises(archive_path):
    """A non-existent (type, event, field) combination fails loudly rather than silently."""
    archive = LightcurveArchive(archive_path)
    with pytest.raises(OSError):
        archive.table("supernovae", "does_not_exist", "L_bol")


def test_default_path_points_at_the_packaged_archive():
    """With no argument, the archive resolves to the packaged `test_data/transients/lightcurves.h5`."""
    archive = LightcurveArchive()
    assert archive.path.name == "lightcurves.h5"
    assert archive.path.exists()
