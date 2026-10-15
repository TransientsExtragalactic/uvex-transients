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
    assert len(archive.objects()) == 336
    assert archive.objects(sn_type="IIP+Excess") == ["SN2023ixf", "SN2024ggi"]
    lc = archive.lightcurve("Gaia16apd", "UVM2")
    assert len(lc) == 12
    assert lc.colnames == ["t", "L", "L_err", "mjd"]


def test_packaged_archive_hinkle_slsne():
    """The recalibrated Hinkle+2023 SLSNe-I are present, detection-only, and timed from TNS discovery."""
    archive = SwiftSNeArchive()
    assert archive.meta("SN2018ibb")["t0_source"] == "discovery_TNS"
    assert archive.meta("LSQ14mo")["t0_source"] == "first_swift"
    lc = archive.lightcurve("SN2018ibb", "UVW1")
    assert len(lc) == archive.meta("SN2018ibb", "UVW1")["n_det"] == 20
    assert (lc["L"] > 0).all()
    assert lc["t"].value == pytest.approx(lc["mjd"].value - archive.meta("SN2018ibb")["t0_mjd"])
    assert "DES15S2nr" not in archive.objects()  # no UV detections
