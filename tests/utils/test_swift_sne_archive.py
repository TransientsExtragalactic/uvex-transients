"""Tests for `uvex_transients.utils.swift_sne_archive.SwiftSNeArchive`."""

import h5py
import pytest
from astropy import units as u

from uvex_transients.utils.swift_sne_archive import SwiftSNeArchive


@pytest.fixture
def archive_path(tmp_path):
    """Build a small synthetic archive mirroring the real ``/<object>/<band>/{t,L,L_err,mjd}`` schema."""
    path = tmp_path / "swift.h5"
    with h5py.File(path, "w") as f:
        for name, sn_type in [("SN2000a", "IIP"), ("SN2000b", "Ia")]:
            f.create_group(name).attrs["type_standard"] = sn_type
        for band, lam in [("UVW1", 2600.0), ("UVM2", 2246.0)]:
            g = f["SN2000a"].create_group(band)
            g.attrs["lambda_eff_AA"] = lam
            g["t"], g["L"], g["L_err"], g["mjd"] = [1.0, 2.0], [1e41, 2e41], [1e40, 1e40], [58000.0, 58001.0]
    return path


def test_objects_and_type_filter(archive_path):
    archive = SwiftSNeArchive(archive_path)
    assert archive.objects() == ["SN2000a", "SN2000b"]
    assert archive.objects(sn_type="IIP") == ["SN2000a"]


def test_bands_and_meta(archive_path):
    archive = SwiftSNeArchive(archive_path)
    assert archive.bands("SN2000a") == ["UVM2", "UVW1"]
    assert archive.meta("SN2000a")["type_standard"] == "IIP"
    assert archive.meta("SN2000a", "UVW1")["lambda_eff_AA"] == 2600.0


def test_lightcurve_columns_and_units(archive_path):
    lc = SwiftSNeArchive(archive_path).lightcurve("SN2000a", "UVW1")
    assert lc.colnames == ["t", "L", "L_err", "mjd"]
    assert lc["t"].unit == u.day
    assert lc["L"].unit == u.erg / u.s
    assert lc["L"].to_value(u.erg / u.s) == pytest.approx([1e41, 2e41])
    assert lc.meta["lambda_eff_AA"] == 2600.0


def test_missing_object_raises(archive_path):
    with pytest.raises(KeyError):
        SwiftSNeArchive(archive_path).lightcurve("nope", "UVW1")


def test_packaged_archive():
    """The packaged archive is detection-only and carries the expected columns."""
    archive = SwiftSNeArchive()
    assert len(archive.objects()) == 311
    lc = archive.lightcurve("Gaia16apd", "UVM2")
    assert len(lc) == 12
    assert lc.colnames == ["t", "L", "L_err", "mjd"]
