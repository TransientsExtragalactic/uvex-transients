"""Tests for `uvex_transients.utils.summary_report`, on a small hand-built event summary table."""

import matplotlib as mpl

mpl.use("Agg")

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import QTable
from astropy.time import Time
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from uvex_transients.utils import summary_report
from uvex_transients.utils.summary_report import UVEX_REGIONS, PopulationSpec, SummaryReport

T0 = Time("2030-01-01T00:00:00")
POPULATIONS = {
    "tde": PopulationSpec("TDE", max_gap=30.0),
    "kilonova": PopulationSpec("Kilonova", max_gap=3.0, color="#123456"),
}

# One row per event: (type, n_det, peak_mag, days from explosion to first detection, days from the last
# constraining non-detection to the first detection or None if there is none).
ROWS = [
    ("tde", 5, 21.0, 4.0, 10.0),  # selected
    ("tde", 3, 22.0, 6.0, 25.0),  # selected
    ("tde", 1, 20.0, 2.0, 5.0),  # one detection only
    ("tde", 4, 23.5, 3.0, 5.0),  # too faint
    ("tde", 4, 22.0, 3.0, 45.0),  # explosion not bracketed tightly enough
    ("tde", 4, 22.0, 3.0, None),  # no constraining non-detection at all
    ("kilonova", 2, 20.5, 1.0, 2.0),  # selected
    ("kilonova", 2, 20.5, 1.0, 4.0),  # gap fails the kilonova's own limit, which a TDE's would pass
    ("ia_sne", 9, 19.0, 1.0, 1.0),  # a population that is not being reported
]


def make_summary(rows=ROWS) -> QTable:
    """Build an event summary table shaped like `run_event_summary_action`'s, one row per entry of `rows`."""
    n = len(rows)
    types, n_det, peak, t_first, gap = (list(column) for column in zip(*rows))
    t_det = T0 + np.array(t_first) * u.day
    no_bracket = np.array([g is None for g in gap])
    t_nondet = Time(
        np.ma.array(
            (t_det.jd - np.array([0.0 if g is None else g for g in gap])),
            mask=no_bracket,
        ),
        format="jd",
    )
    table = QTable(
        {
            "event_id": np.arange(n),
            "transient_type": np.array(types),
            "redshift": np.linspace(0.05, 0.3, n),
            "coord": SkyCoord(np.linspace(10, 300, n) * u.deg, np.linspace(-60, 60, n) * u.deg),
            "weight": np.where(np.array(types) == "tde", 10.0, 0.5),
            "n_det": np.array(n_det),
            "t_explosion": Time(np.full(n, T0.jd), format="jd"),
            "t_first_det": t_det,
            "t_last_constraining_nondet": t_nondet,
            "peak_mag": np.array(peak),
        }
    )
    table.meta.update(
        {
            "snr_threshold": 5.0,
            "n_pre_cut": {"tde": 100, "kilonova": 20, "ia_sne": 1000},
            "expected_events": {"tde": 1000.0, "kilonova": 10.0, "ia_sne": 5000.0},
            "rate_ci": {"tde": [0.5, 2.0], "kilonova": [0.8, 1.5], "ia_sne": [0.9, 1.1]},
        }
    )
    return table


@pytest.fixture
def report():
    """Build a report on the table `make_summary` builds."""
    return SummaryReport(make_summary(), POPULATIONS)


@pytest.fixture(autouse=True)
def close_figures():
    """Close every figure a test opened."""
    yield
    plt.close("all")


class TestConstruction:
    """Building a report from a table."""

    def test_only_the_requested_populations_are_reported(self, report):
        assert list(report.populations) == ["tde", "kilonova"]

    def test_a_population_the_table_did_not_generate_is_dropped(self):
        populations = {**POPULATIONS, "lfbot": PopulationSpec("LFBOT", max_gap=5.0)}
        assert "lfbot" not in SummaryReport(make_summary(), populations).populations

    def test_no_requested_population_in_the_table_raises(self):
        with pytest.raises(LookupError, match="none of the populations"):
            SummaryReport(make_summary(), {"lfbot": PopulationSpec("LFBOT", max_gap=5.0)})

    def test_missing_column_raises(self):
        table = make_summary()
        table.remove_column("peak_mag")
        with pytest.raises(ValueError, match="peak_mag"):
            SummaryReport(table, POPULATIONS)

    def test_missing_meta_raises(self):
        table = make_summary()
        del table.meta["rate_ci"]
        with pytest.raises(ValueError, match="rate_ci"):
            SummaryReport(table, POPULATIONS)

    def test_colors_default_in_order_and_explicit_ones_are_kept(self, report):
        assert report.populations["kilonova"].color == "#123456"
        assert report.populations["tde"].color.startswith("#")

    def test_snr_threshold_comes_from_the_table(self, report):
        assert report.snr_threshold == 5.0
        assert SummaryReport(make_summary(), POPULATIONS, snr_threshold=7.0).snr_threshold == 7.0


class TestSelection:
    """The three cumulative cuts."""

    def test_each_cut_is_cumulative(self, report):
        assert report.repeated.sum() == 8
        assert report.bright.sum() == 7
        assert report.selected.sum() == 3
        assert np.all(report.selected <= report.bright) and np.all(report.bright <= report.repeated)

    def test_selected_events(self, report):
        assert report.selected.tolist() == [True, True, False, False, False, False, True, False, False]

    def test_each_population_uses_its_own_gap_limit(self, report):
        # Both kilonovae are detected twice and bright, and differ only in their bracketing gap.
        assert report.selected[6] and not report.selected[7]

    def test_no_bracket_fails_instead_of_slipping_through(self, report):
        assert np.isnan(report.gap[5])
        assert not report.selected[5]

    def test_unreported_population_is_never_selected(self, report):
        assert not report.selected[8]

    def test_cut_values_are_configurable(self):
        loose = SummaryReport(make_summary(), POPULATIONS, min_detections=1, peak_mag_limit=24.0)
        assert loose.selected[2] and loose.selected[3]


class TestCountsAndYields:
    """The funnel counts and the expected yields."""

    def test_stage_counts_funnel_down_from_the_generated_number(self, report):
        counts = report.stage_counts()
        assert counts["tde"] == [100, 6, 5, 4, 2]
        assert counts["kilonova"] == [20, 2, 2, 2, 1]
        assert len(counts["tde"]) == len(SummaryReport.STAGE_NAMES)

    def test_yields_match_estimate_yield(self, report):
        yields = report.yields()
        assert sorted(yields["transient_type"]) == ["kilonova", "tde"]
        tde = yields[yields["transient_type"] == "tde"][0]
        assert tde["n_selected"] == 2
        assert tde["expected_events"] == pytest.approx(1000.0 * 2 / 100)

    def test_yield_table_has_one_row_per_population(self, report):
        table = report.yield_table()
        assert list(table["Population"]) == ["TDE", "Kilonova"]
        assert table["Selected / generated"][0] == "2 / 100"

    def test_a_population_with_nothing_selected_gets_an_upper_limit(self):
        rows = [row for row in ROWS if row[0] != "kilonova"] + [("kilonova", 1, 20.0, 1.0, 1.0)]
        table = SummaryReport(make_summary(rows), POPULATIONS).yield_table()
        assert table["Expected selected events"][1].startswith("<")


class TestFigures:
    """The figures."""

    def test_funnel(self, report):
        fig = report.plot_funnel()
        assert isinstance(fig, Figure)
        assert fig.axes[0].get_yscale() == "log"

    def test_distributions_for_several_populations_are_one_figure_per_quantity(self, report):
        figs = report.plot_distributions()
        assert len(figs) == 5
        assert all(len(fig.axes) == 2 for fig in figs)

    def test_distributions_for_one_population_are_a_single_figure(self):
        figs = SummaryReport(make_summary(), {"tde": POPULATIONS["tde"]}).plot_distributions()
        assert len(figs) == 1

    def test_a_population_with_no_usable_values_draws_a_placeholder(self):
        rows = [row for row in ROWS if row[0] != "kilonova"] + [("kilonova", 1, 20.0, 1.0, None)]
        figs = SummaryReport(make_summary(rows), POPULATIONS).plot_distributions()
        gap_axes = figs[-1].axes[1]
        assert [text.get_text() for text in gap_axes.texts] == ["no events"]

    def test_sky(self, report):
        fig = report.plot_sky(footprints={})
        assert isinstance(fig, Figure)

    def test_sky_shades_each_region_and_keeps_the_event_legend(self, report):
        from mocpy import MOC

        regions = {
            "Cap": MOC.from_cone(lon=0 * u.deg, lat=0 * u.deg, radius=20 * u.deg, max_depth=5),
            "Spot": MOC.from_cone(lon=90 * u.deg, lat=30 * u.deg, radius=5 * u.deg, max_depth=5),
        }
        fig = report.plot_sky(footprints=regions)
        labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
        assert labels[0].startswith("Cap (") and labels[1].startswith("Spot (")
        assert any(label.startswith("Detected") for label in labels)

    def test_sky_defaults_to_the_three_uvex_regions(self, report, monkeypatch):
        from mocpy import MOC

        from uvex_transients.surveys.footprints import default_registry

        moc = MOC.from_cone(lon=0 * u.deg, lat=0 * u.deg, radius=10 * u.deg, max_depth=5)
        monkeypatch.setattr(default_registry, "__getitem__", lambda self, name: moc)
        fig = report.plot_sky()
        labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
        assert [label.split(" (")[0] for label in labels[:3]] == list(UVEX_REGIONS)

    def test_example_event_needs_a_selected_event(self):
        rows = [row for row in ROWS if row[0] != "kilonova"] + [("kilonova", 1, 20.0, 1.0, 1.0)]
        report = SummaryReport(make_summary(rows), POPULATIONS)
        with pytest.raises(ValueError, match="No Kilonova events"):
            report.plot_example_event("kilonova", None, {}, None, None)


class TestFromRelease:
    """Building a report from a published release."""

    def test_reads_the_summary_asset(self, tmp_path, monkeypatch):
        path = tmp_path / "event_summary.ecsv"
        make_summary().write(path)
        calls = []

        def fake_get_results(assets, tag=None):
            calls.append((assets, tag))
            return {"summary": path}

        monkeypatch.setattr(summary_report, "get_results", fake_get_results)
        report = SummaryReport.from_release(POPULATIONS, tag="v9", peak_mag_limit=24.0)
        assert calls == [(["summary"], "v9")]
        assert report.peak_mag_limit == 24.0
        assert list(report.populations) == ["tde", "kilonova"]

    @pytest.mark.parametrize("error", [ConnectionError, LookupError])
    def test_an_unavailable_release_propagates_for_the_caller_to_skip(self, monkeypatch, error):
        def fail(assets, tag=None):
            raise error("no release")

        monkeypatch.setattr(summary_report, "get_results", fail)
        with pytest.raises(error):
            SummaryReport.from_release(POPULATIONS)
