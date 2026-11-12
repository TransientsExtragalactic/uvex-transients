"""Tests for the stage commands of ``phase-space`` (run, fit, plot, compare, info, kinds) and `phase_space.stages`."""

import shutil
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table
from click.testing import CliRunner

pytest.importorskip("sklearn")
pytest.importorskip("matplotlib")

from uvex_transients.cli.main import cli  # noqa: E402
from uvex_transients.phase_space import stages  # noqa: E402

from .test_phase_space import BREAKOUT_MAP, CONFIG_TEMPLATE  # noqa: E402

MAPS = ("fuv_test", "breakout_test")

#: A points file of the earlier format, as the releases made before the evaluator was removed hold them.
EARLIER = Path(__file__).parents[1] / "phase_space" / "data" / "legacy_points.ecsv"


def invoke(*args, ok=True):
    result = CliRunner().invoke(cli, ["phase-space", *map(str, args)])
    if ok:
        assert result.exit_code == 0, result.output
    return result


def same_table(a, b, rtol=0.0):
    """Assert that two ECSV tables hold the same columns, exactly or (for the floats) to a relative tolerance."""
    a, b = Table.read(a, format="ascii.ecsv"), Table.read(b, format="ascii.ecsv")
    assert a.colnames == b.colnames and len(a) == len(b)
    for name in a.colnames:
        x, y = np.asarray(a[name]), np.asarray(b[name])
        if rtol and x.dtype.kind == "f":
            np.testing.assert_allclose(x, y, rtol=rtol, atol=1e-300, err_msg=name)
        else:
            np.testing.assert_array_equal(x, y, err_msg=name)


#: The Gaussian process is fitted again from the stored points, where a run fits it as the points arrive, so the two
#: optimizers end at marginally different optima. On these four-point fits the maps agree to between 1e-7 (the recovery
#: fraction) and 1e-5 (its posterior width), and this leaves room above that.
FIT_RTOL = 1e-3


@pytest.fixture(scope="module")
def run_config(tmp_path_factory, make_schedule):
    """A run-config over a synthetic schedule with a canonical and a breakout map."""
    directory = tmp_path_factory.mktemp("stages")
    schedule_path, fov_path = directory / "schedule.ecsv", directory / "schedule.reg"
    make_schedule().to_disk(schedule_path, fov_path=fov_path)
    path = directory / "phase_space.yaml"
    path.write_text(CONFIG_TEMPLATE.format(schedule_path=schedule_path, fov_path=fov_path) + BREAKOUT_MAP)
    return path


@pytest.fixture(scope="module")
def built(run_config, tmp_path_factory):
    """The maps made in one pass, by ``build``."""
    out = tmp_path_factory.mktemp("built")
    invoke(run_config, "--out-dir", out, *(arg for map_id in (*MAPS, "nuv_test") for arg in ("--map", map_id)))
    return out


@pytest.fixture(scope="module")
def staged(run_config, tmp_path_factory):
    """The same maps made in stages: ``run`` on two workers, then ``fit`` and ``plot``."""
    out = tmp_path_factory.mktemp("staged")
    invoke("run", run_config, "--out-dir", out, "--workers", 2, *(arg for map_id in MAPS for arg in ("--map", map_id)))
    return out


@pytest.fixture(scope="module")
def fitted(run_config, staged):
    """The staged maps, fitted and drawn from their points."""
    invoke("fit", staged, "--config", run_config, "--grid-size", 5)
    invoke("plot", staged)
    return staged


# ---------------------------------------------------------------------------- #
# run                                                                          #
# ---------------------------------------------------------------------------- #
def test_run_writes_only_the_points(staged):
    for map_id in MAPS:
        assert [path.name for path in (staged / map_id).iterdir()] == ["points.ecsv"]


def test_the_points_do_not_depend_on_the_stages_or_the_workers(built, staged):
    """The breakout points of a run on two workers are those of the serial build, point for point."""
    same_table(built / "breakout_test" / "points.ecsv", staged / "breakout_test" / "points.ecsv")


def test_the_canonical_points_are_those_of_a_build_whatever_the_workers(built, staged):
    """The staged run measured on two workers, and each point has its own seed, so the points are the serial ones."""
    same_table(built / "fuv_test" / "points.ecsv", staged / "fuv_test" / "points.ecsv")


# ---------------------------------------------------------------------------- #
# fit and plot                                                                 #
# ---------------------------------------------------------------------------- #
def test_fit_reproduces_the_canonical_files_of_a_build(built, fitted):
    same_table(built / "fuv_test" / "map.ecsv", fitted / "fuv_test" / "map.ecsv", FIT_RTOL)
    same_table(built / "fuv_test" / "overlay.ecsv", fitted / "fuv_test" / "overlay.ecsv")


def test_fit_reproduces_the_breakout_map_of_a_build(built, fitted):
    same_table(built / "breakout_test" / "map.ecsv", fitted / "breakout_test" / "map.ecsv", FIT_RTOL)


def test_an_overlay_the_run_config_asks_for_is_made_for_the_canonical_map_only(fitted):
    """The config asks for an overlay on every map, but a breakout map has none, and that is not an error."""
    assert (fitted / "fuv_test" / "overlay.ecsv").exists()
    assert not (fitted / "breakout_test" / "overlay.ecsv").exists()


def test_fit_refuses_to_replace_a_map_without_overwrite(run_config, fitted):
    result = invoke("fit", fitted / "breakout_test", "--config", run_config, ok=False)
    assert result.exit_code == 1 and "overwrite" in result.output
    invoke("fit", fitted / "breakout_test", "--config", run_config, "--grid-size", 5, "--overwrite")


def test_fit_without_a_run_config_needs_a_schedule_it_can_fetch(fitted):
    """The schedule here is a file, which the points cannot name, so a run-config has to supply it."""
    result = invoke("fit", fitted / "breakout_test", "--overwrite", ok=False)
    assert result.exit_code != 0 and "schedule" in (result.output + str(result.exception)).lower()


def test_a_directory_of_maps_stands_for_the_maps_in_it(run_config, fitted):
    result = invoke("fit", fitted, "--config", run_config, "--grid-size", 5, "--overwrite")
    assert "fuv_test" in result.output and "breakout_test" in result.output


def test_plot_draws_both_kinds_from_their_files(fitted):
    for map_id in MAPS:
        (fitted / map_id / "map.png").unlink(missing_ok=True)
    invoke("plot", fitted)
    for map_id in MAPS:
        assert (fitted / map_id / "map.png").stat().st_size > 0


def test_plot_can_write_elsewhere_for_one_map_only(fitted, tmp_path):
    figure = tmp_path / "elsewhere.png"
    invoke("plot", fitted / "breakout_test", "--out", figure, "--title", "a title", "--show-points")
    assert figure.stat().st_size > 0
    result = invoke("plot", fitted, "--out", figure, ok=False)
    assert result.exit_code == 1 and "one map" in result.output


def test_plot_needs_a_fitted_map(run_config, tmp_path):
    invoke("run", run_config, "--out-dir", tmp_path, "--map", "breakout_test")
    result = invoke("plot", tmp_path / "breakout_test", ok=False)
    assert result.exit_code == 1 and "fit" in result.output


# ---------------------------------------------------------------------------- #
# Points files of the earlier format                                           #
# ---------------------------------------------------------------------------- #
@pytest.fixture
def earlier(tmp_path):
    """A map directory holding only a points file of the earlier format."""
    directory = tmp_path / "earlier"
    directory.mkdir()
    shutil.copy(EARLIER, directory / "points.ecsv")
    return directory


def test_info_says_that_a_points_file_is_of_the_earlier_format(earlier):
    assert "kind      canonical (earlier points format)" in invoke("info", earlier).output


def test_a_points_file_of_the_earlier_format_can_be_fitted_and_drawn(run_config, earlier):
    """The files of a published release are read as the canonical experiment, with an overlay if one is asked for."""
    invoke("fit", earlier, "--config", run_config, "--grid-size", 4, "--overlay", "--overlay-draws", 5)
    invoke("plot", earlier)
    for name in ("map.ecsv", "overlay.ecsv", "map.png"):
        assert (earlier / name).stat().st_size > 0


# ---------------------------------------------------------------------------- #
# compare                                                                      #
# ---------------------------------------------------------------------------- #
def test_compare_a_map_with_itself_is_no_difference(built, tmp_path):
    invoke("compare", built / "breakout_test", built / "breakout_test", "--out-dir", tmp_path)
    difference = Table.read(tmp_path / "difference.ecsv", format="ascii.ecsv")
    assert difference.colnames[:2] == ["log_L0", "log_t0"]
    assert np.all(np.isnan(difference["delta_dex"]) | (difference["delta_dex"] == 0))
    assert (tmp_path / "difference.png").stat().st_size > 0


def test_compare_refuses_maps_of_other_bands_unless_told_not_to(built, tmp_path):
    """The canonical maps are the FUV and the NUV, whose luminosities are not in the same band."""
    args = ["compare", built / "fuv_test", built / "nuv_test", "--out-dir", tmp_path]
    result = invoke(*args, ok=False)
    assert result.exit_code == 1 and "band" in result.output
    invoke(*args, "--no-check-meta", "--no-plot")
    assert (tmp_path / "difference.ecsv").exists() and not (tmp_path / "difference.png").exists()


def test_compare_maps_of_different_kinds_are_refused(built, tmp_path):
    result = invoke("compare", built / "fuv_test", built / "breakout_test", "--out-dir", tmp_path, ok=False)
    assert result.exit_code == 1 and "coordinates" in result.output


def test_compare_does_not_overwrite_without_being_told_to(built, tmp_path):
    args = ["compare", built / "breakout_test", built / "breakout_test", "--out-dir", tmp_path]
    invoke(*args)
    assert invoke(*args, ok=False).exit_code == 1
    invoke(*args, "--overwrite", "--no-plot")


# ---------------------------------------------------------------------------- #
# info and kinds                                                               #
# ---------------------------------------------------------------------------- #
def test_info_says_what_a_map_is(built):
    out = invoke("info", built).output
    assert "kind      canonical\n" in out and "kind      breakout" in out and "earlier" not in out
    assert "axis      log_L0: " in out and "axis      log_luminosity: " in out
    assert "files     points, map, overlay, figure" in out or "files     points, map, figure" in out


def test_info_of_a_path_that_is_not_a_map_is_an_error(tmp_path):
    result = invoke("info", tmp_path, ok=False)
    assert result.exit_code == 1 and "is not a map" in result.output


def test_kinds_lists_each_sampler_and_what_it_takes():
    out = invoke("kinds").output
    assert "breakout:" in out and "canonical:" in out and "takes:" in out and "z_max" in out


# ---------------------------------------------------------------------------- #
# The library                                                                  #
# ---------------------------------------------------------------------------- #
def test_describe_reads_both_formats(built, earlier):
    old, current = stages.describe(earlier), stages.describe(built / "fuv_test")
    new = stages.describe(built / "breakout_test")
    assert (old.flavor, old.kind, old.band) == ("legacy", "canonical", "FUV")
    assert (current.flavor, current.kind, current.band) == ("experiment", "canonical", "FUV")
    assert (new.flavor, new.kind, new.band) == ("experiment", "breakout", "FUV")
    assert old.n_points == 6 and current.n_points == new.n_points == 4 and old.n_tries > 0
    assert new.space.names == ("log_L0", "log_t0")


def test_find_maps_expands_a_run_and_refuses_what_is_not_a_map(built, tmp_path):
    assert [path.name for path in stages.find_maps([built])] == ["breakout_test", "fuv_test", "nuv_test"]
    assert stages.find_maps([built / "fuv_test" / "points.ecsv"]) == [built / "fuv_test"]
    with pytest.raises(FileNotFoundError, match="is not a map"):
        stages.find_maps([tmp_path])
    with pytest.raises(FileNotFoundError, match="does not exist"):
        stages.find_maps([tmp_path / "nothing"])


def test_an_overlay_is_refused_for_a_map_that_is_not_over_the_canonical_axes(built, tmp_path):
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(built / "breakout_test", copy)
    with pytest.raises(ValueError, match="overlay"):
        stages.fit_map(copy, overlay=True, overwrite=True)
