"""Tests for `uvex_transients.utils.luminosity_function_archive` and the script that builds the archive."""

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy import units as u

from uvex_transients.utils.luminosity_function_archive import LuminosityFunctionArchive

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "build_luminosity_functions", _ROOT / "scripts" / "build_luminosity_functions.py"
)
build_luminosity_functions = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_luminosity_functions)

TEXT = """# X axis is M_V,peak
# Y axis is Rate SN / yr / Mpc^3 / mag^-1 h^370

# ZTF-BTS ccSNe
x = [-16.0, -18.0, -17.0]
y = [1.0, 3.0, 2.0]

# ASAS-SN ccSNe
x = [-15.0, -16.0]
y = [5.0, 6.0]

# ASAS-SN SNe Ia
x = [-19.0]
y = [7.0]
"""


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "SNe_LFs"
    path.write_text(TEXT)
    return path


@pytest.fixture
def archive(source, tmp_path):
    return LuminosityFunctionArchive(build_luminosity_functions.build(source, tmp_path / "lf.h5"))


class TestParse:
    def test_curves_come_back_in_file_order_with_the_axis_header(self):
        header, curves = build_luminosity_functions.parse(TEXT)
        assert header == ["X axis is M_V,peak", "Y axis is Rate SN / yr / Mpc^3 / mag^-1 h^370"]
        assert list(curves) == ["ZTF-BTS ccSNe", "ASAS-SN ccSNe", "ASAS-SN SNe Ia"]
        np.testing.assert_array_equal(curves["ZTF-BTS ccSNe"][0], [-16.0, -18.0, -17.0])

    @pytest.mark.parametrize("text", ["# A\nx = [1.0, 2.0]\ny = [1.0]\n", "# A\ny = [1.0]\n", "x = [1.0]\ny = [1.0]\n"])
    def test_a_curve_without_matching_lists_is_rejected(self, text):
        with pytest.raises(ValueError):
            build_luminosity_functions.parse(text)


class TestBuild:
    def test_the_layout_is_type_then_band_then_source(self, archive):
        with h5py.File(archive.path, "r") as f:
            assert set(f["supernovae/ccSNe/luminosity_function_V"]) == {"ASAS-SN", "ZTF-BTS"}
            assert set(f["supernovae/Ia/luminosity_function_V/ASAS-SN"]) == {"M", "dAndM"}

    def test_a_label_that_has_no_place_is_rejected(self, tmp_path):
        source = tmp_path / "SNe_LFs"
        source.write_text("# Mystery SNe\nx = [1.0]\ny = [1.0]\n")
        with pytest.raises(KeyError, match="Mystery"):
            build_luminosity_functions.build(source, tmp_path / "lf.h5")

    def test_magnitudes_are_stored_ascending_with_their_rates(self, archive):
        table = archive.table("supernovae/ccSNe", "ZTF-BTS")
        np.testing.assert_array_equal(table["M"].value, [-18.0, -17.0, -16.0])
        np.testing.assert_array_equal(table["dAndM"].value, [3.0, 2.0, 1.0])

    def test_the_packaged_archive_is_what_the_source_file_builds(self, tmp_path):
        rebuilt = build_luminosity_functions.build(build_luminosity_functions.SOURCE, tmp_path / "lf.h5")
        packaged, fresh = LuminosityFunctionArchive(), LuminosityFunctionArchive(rebuilt)
        assert packaged.types() == fresh.types()
        for transient_type in fresh.types():
            for source in fresh.sources(transient_type):
                np.testing.assert_array_equal(
                    packaged.table(transient_type, source)["dAndM"].value,
                    fresh.table(transient_type, source)["dAndM"].value,
                )


class TestArchive:
    def test_types_and_sources(self, archive):
        assert archive.types() == ["supernovae/Ia", "supernovae/ccSNe"]
        assert archive.sources("supernovae/ccSNe") == ["ASAS-SN", "ZTF-BTS"]
        assert archive.types(band="B") == []

    def test_a_table_has_units_and_the_source_attributes(self, archive):
        table = archive.table("supernovae/Ia")
        assert table["M"].unit == u.mag and table["dAndM"].unit == u.Mpc**-3 / u.yr / u.mag
        assert table.meta["label"] == "ASAS-SN SNe Ia" and "M_V,peak" in table.meta["axis_header"]

    def test_the_source_may_be_left_out_only_when_there_is_one(self, archive):
        assert len(archive.table("supernovae/Ia")) == 1
        with pytest.raises(ValueError, match="several sources"):
            archive.table("supernovae/ccSNe")

    def test_repr_names_the_types_and_survives_a_missing_file(self, archive, tmp_path):
        assert "supernovae/Ia" in repr(archive)
        assert "not found" in repr(LuminosityFunctionArchive(tmp_path / "none.h5"))
