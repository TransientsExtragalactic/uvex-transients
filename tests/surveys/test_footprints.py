"""Tests for `uvex_transients.surveys.footprints`: registry, MOC caching, shape generators, and LSST gating."""

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from regions import PointSkyRegion, Regions

from uvex_transients.surveys.footprints import base, lsst
from uvex_transients.surveys.footprints.base import FootprintRegistry, SurveyFootprint, combine_footprints
from uvex_transients.surveys.footprints.utils import (
    convert_region_to_MOC,
    dec_band_MOC,
    galactic_box_MOC,
    union_of_galactic_boxes_MOC,
)


@pytest.fixture
def registry(monkeypatch, tmp_path):
    """A fresh default registry and an empty cache directory, so tests never touch the real ones."""
    fresh = FootprintRegistry()
    monkeypatch.setattr(base, "default_registry", fresh)
    monkeypatch.setattr(base, "cache_dir", tmp_path)
    return fresh


def _counting_footprint(name="test:band", **kwargs):
    """A cached `dec_band_MOC` footprint whose generator call count is exposed as ``.calls``."""
    calls = []

    def generator(**params):
        calls.append(params)
        return dec_band_MOC(**params)

    footprint = SurveyFootprint(
        name=name, generator=generator, params={"min_dec": 0.0}, persist=True, MOC_max_order=6, **kwargs
    )
    footprint.calls = calls
    return footprint


class TestRegistry:
    def test_lookup_is_case_insensitive(self, registry):
        footprint = _counting_footprint("Test:Band")
        assert registry["test:band"] is footprint
        assert "TEST:BAND" in registry

    def test_duplicate_name_raises_unless_overwritten(self, registry):
        footprint = _counting_footprint()
        with pytest.raises(KeyError, match="already registered"):
            footprint.register()
        footprint.register(overwrite=True)

    def test_unknown_name_lists_known_names(self, registry):
        _counting_footprint()
        with pytest.raises(KeyError, match="test:band"):
            registry["nope"]

    def test_names_filters_by_prefix(self, registry):
        _counting_footprint("a:one")
        _counting_footprint("b:two")
        assert registry.names("a:") == ["a:one"]

    def test_match_returns_one_mask_per_name(self, registry):
        _counting_footprint()
        masks = registry.match([10.0, 10.0], [45.0, -45.0], ["test:band"])
        assert masks["test:band"].tolist() == [True, False]


def _band(name, min_dec, max_dec):
    return SurveyFootprint(
        name=name, generator=dec_band_MOC, params={"min_dec": min_dec, "max_dec": max_dec}, MOC_max_order=6
    )


class TestCombine:
    @pytest.fixture
    def bands(self, registry):
        return _band("t:a", 0.0, 40.0), _band("t:b", 20.0, 60.0)

    @pytest.mark.parametrize(
        "operation, inside, outside",
        [
            ("union", [10, 30, 50], [-10, 70]),
            ("intersection", [30], [10, 50]),
            ("difference", [10], [30, 50]),
            ("xor", [10, 50], [30, -10]),
        ],
    )
    def test_operations(self, bands, operation, inside, outside):
        combined = combine_footprints("t:c", operation, bands, persist=False)
        assert combined.contains(np.zeros(len(inside)) + 5, inside).all()
        assert not combined.contains(np.zeros(len(outside)) + 5, outside).any()

    def test_accepts_names_and_registers(self, bands, registry):
        combined = combine_footprints("t:c", "or", ["t:a", "t:b"], persist=False)
        assert registry["t:c"] is combined

    def test_operand_change_invalidates_cache_key(self, bands, registry):
        before = combine_footprints("t:c", "union", bands).cache_key
        registry["t:a"].version = "v2"
        registry._footprints.pop("t:c")
        assert combine_footprints("t:c", "union", bands).cache_key != before

    def test_bad_inputs_raise(self, bands):
        with pytest.raises(ValueError, match="Unknown operation"):
            combine_footprints("t:c", "nand", bands)
        with pytest.raises(ValueError, match="At least one"):
            combine_footprints("t:c", "union", [])
        with pytest.raises(KeyError):
            combine_footprints("t:c", "union", ["t:a", "t:missing"])


class TestCaching:
    def test_second_instance_loads_from_disk_without_generating(self, registry):
        first = _counting_footprint()
        first.moc
        assert len(first.calls) == 1
        assert first.is_cached

        registry._footprints.clear()
        second = _counting_footprint()
        second.moc
        assert second.calls == []

    def test_clear_from_cache_forces_regeneration(self, registry):
        footprint = _counting_footprint()
        footprint.moc
        footprint.clear_from_cache()
        assert not footprint.is_cached
        footprint.moc
        assert len(footprint.calls) == 2

    def test_cache_key_tracks_params_and_version(self, registry):
        base_key = _counting_footprint().cache_key
        assert _counting_footprint("test:v", version="v2").cache_key != base_key
        other = _counting_footprint("test:band2")
        other.params = {"min_dec": 10.0}
        assert other.cache_key != base_key

    def test_generator_must_return_a_moc(self, registry):
        footprint = SurveyFootprint(name="test:bad", generator=lambda **_: "not a moc")
        with pytest.raises(TypeError, match="not a MOC"):
            footprint.moc


class TestGenerators:
    def test_dec_band_contains_only_its_band(self):
        moc = dec_band_MOC(max_order=6, min_dec=0.0)
        coords = SkyCoord([10, 10] * u.deg, [45, -45] * u.deg)
        assert moc.contains_skycoords(coords).tolist() == [True, False]

    def test_galactic_box_contains_its_center(self):
        moc = galactic_box_MOC(max_order=6, l_min=-10, l_max=10, b_min=-5, b_max=5)
        inside = SkyCoord(l=0 * u.deg, b=0 * u.deg, frame="galactic")
        outside = SkyCoord(l=90 * u.deg, b=0 * u.deg, frame="galactic")
        assert moc.contains_skycoords(inside)
        assert not moc.contains_skycoords(outside)

    def test_empty_inputs_raise(self):
        with pytest.raises(ValueError, match="boxes"):
            union_of_galactic_boxes_MOC(max_order=6, boxes=[])
        with pytest.raises(ValueError, match="regions"):
            convert_region_to_MOC(Regions([]), max_order=6)

    def test_point_region_needs_a_size(self):
        regions = Regions([PointSkyRegion(SkyCoord(10 * u.deg, 10 * u.deg))])
        with pytest.raises(ValueError, match="point_region_size"):
            convert_region_to_MOC(regions, max_order=6)
        moc = convert_region_to_MOC(regions, max_order=6, point_region_size=1.0)
        assert moc.contains_skycoords(SkyCoord(10 * u.deg, 10 * u.deg))


class TestLSST:
    def test_footprints_are_registered_without_rubin_scheduler(self):
        names = base.default_registry.names("lsst:")
        assert set(names) == {fp.name for fp in lsst.lsst_footprints.values()} | {
            lsst.lsst_ddf_footprint.name,
            lsst.lsst_combined_footprint.name,
        }

    def test_generation_without_rubin_scheduler_explains_how_to_install(self, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "rubin_scheduler", None)
        with pytest.raises(ImportError, match=r"uvex-transients\[rubin\]"):
            lsst.lsst_ddf_MOC(max_order=6, radius=1.75)
        with pytest.raises(ImportError, match=r"uvex-transients\[rubin\]"):
            lsst.lsst_regions_MOC(max_order=6, labels=[], nside=16)

    def test_region_labels_become_a_moc(self, monkeypatch):
        """Selecting labels from a stub sky-area map yields a MOC covering exactly those pixels."""
        nside = 4
        labels = np.array([""] * (12 * nside**2), dtype="<U20")
        labels[:10] = "wfd"
        lsst._pixel_labels.cache_clear()
        monkeypatch.setattr(lsst, "_pixel_labels", lambda _nside: labels)

        moc = lsst.lsst_regions_MOC(max_order=6, labels=["wfd"], nside=nside)
        everything = lsst.lsst_regions_MOC(max_order=6, labels=[], nside=nside)

        assert moc.sky_fraction == pytest.approx(10 / (12 * nside**2))
        assert everything.sky_fraction == moc.sky_fraction
