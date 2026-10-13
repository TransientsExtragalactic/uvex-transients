"""Tests for the docs report generator, `docs/source/_report_gen.py`, with the release fetch mocked out."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.table import QTable

from uvex_transients.simulation.rates import estimate_yield

_SOURCE = Path(__file__).resolve().parents[2] / "docs" / "source"
sys.path.insert(0, str(_SOURCE))  # `_report_gen` imports its sibling `_report_runs`, as it does under Sphinx

_spec = importlib.util.spec_from_file_location("_report_gen", _SOURCE / "_report_gen.py")
report_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report_gen)


@pytest.fixture
def summary_file(tmp_path):
    """A small event summary table on disk, as `run_event_summary_action` would write it."""
    table = QTable()
    table["transient_type"] = np.array(["tde"] * 6 + ["slsne"] * 2)
    table.meta.update(
        {
            "n_pre_cut": {"tde": 100, "slsne": 50, "kilonova": 20},
            "expected_events": {"tde": 1000.0, "slsne": 400.0, "kilonova": 30.0},
            "rate_ci": {"tde": [0.5, 2.0], "slsne": [1.0, 1.0], "kilonova": [0.8, 1.5]},
        }
    )
    path = tmp_path / "event_summary.ecsv"
    table.write(path)
    return path


@pytest.fixture
def srcdir(tmp_path):
    """A docs source directory holding two runs: a primary one with two scripts, and a secondary one with none."""
    examples = tmp_path / "report_examples"
    (examples / "main" / "sub").mkdir(parents=True)
    (examples / "main" / "run.yaml").write_text(
        "title: Main Run\nsummary: >\n  The primary\n  run.\nconfig: configs/main.yaml\nicon: telescope\norder: 1\n"
        "tags: [TDEs, SNe]\n"
    )
    (examples / "main" / "plot_a.py").write_text("")
    (examples / "main" / "sub" / "plot_b.py").write_text("")
    (examples / "side").mkdir()
    (examples / "side" / "run.yaml").write_text(
        "title: Side Run\nsummary: A secondary run.\nsimulation: side\norder: 2\n"
    )
    (examples / "not_a_run").mkdir()  # no run.yaml, so not a run
    return tmp_path


def _release(tag="v9"):
    return {"tag_name": tag, "html_url": f"https://example.test/releases/{tag}"}


def test_summary_table_lists_every_type_with_its_display_name(summary_file):
    yields = estimate_yield(QTable.read(summary_file))
    rst = report_gen._summary_table_rst(yields)
    assert "Tidal Disruption Event" in rst
    assert "SLSNe (Magnetar)" in rst
    assert "Kilonova" in rst  # generated but never detected, so it still has a row
    assert rst.count(":math:") == 2 * len(yields)


def test_zero_detections_render_without_nan(summary_file):
    rst = report_gen._summary_table_rst(estimate_yield(QTable.read(summary_file)))
    assert "nan" not in rst.lower()
    assert "--" not in rst


def test_generate_report_writes_the_page_from_the_latest_release(srcdir, monkeypatch, summary_file):
    monkeypatch.setattr(report_gen, "find_release", lambda: _release("v9"))
    monkeypatch.setattr(report_gen, "get_results", lambda assets, tag: {"summary": summary_file})

    report_gen.generate_report(SimpleNamespace(srcdir=srcdir))

    rst = (srcdir / "report" / "index.rst").read_text()
    assert "v9" in rst and "https://example.test/releases/v9" in rst
    assert "Tidal Disruption Event" in rst
    assert "90% Clopper-Pearson" in rst
    assert ".. _report_page:" in rst  # the galleries link back to it


def test_the_page_has_one_card_per_run_in_order(srcdir, monkeypatch, summary_file):
    monkeypatch.setattr(report_gen, "find_release", lambda: _release())
    monkeypatch.setattr(report_gen, "get_results", lambda assets, tag: {"summary": summary_file})

    report_gen.generate_report(SimpleNamespace(srcdir=srcdir))

    rst = (srcdir / "report" / "index.rst").read_text()
    assert rst.count(".. grid-item-card::") == 2  # `not_a_run` has no run.yaml
    assert rst.index("Main Run") < rst.index("Side Run")
    assert ":link: main/index" in rst and ":link: side/index" in rst
    assert "The primary run." in rst  # the folded YAML summary is a single line
    assert ":bdg-secondary-line:`TDEs`" in rst
    assert "2 analyses" in rst  # scripts are counted in subsections too
    assert "Analyses coming soon" in rst  # a run with no scripts yet
    assert rst.count("Primary run") == 1  # only the run without a `simulation`
    assert "configs/main.yaml" in rst
    assert "   :hidden:" in rst and "   main/index" in rst  # the sidebar lists the galleries


@pytest.mark.parametrize("error", [ConnectionError("offline"), LookupError("no such asset"), OSError("disk")])
def test_generate_report_falls_back_to_a_placeholder_on_any_fetch_failure(srcdir, monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(report_gen, "find_release", fail)

    report_gen.generate_report(SimpleNamespace(srcdir=srcdir))

    rst = (srcdir / "report" / "index.rst").read_text()
    assert "No published release report is available yet" in rst
    assert ".. _report_page:" in rst
    assert ":link: main/index" in rst  # the galleries stay reachable without data


def test_generate_report_falls_back_when_the_release_lacks_the_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(report_gen, "find_release", lambda: _release())

    def missing(assets, tag):
        raise LookupError("Release v9 has no asset(s) ['event_summary.ecsv'].")

    monkeypatch.setattr(report_gen, "get_results", missing)

    report_gen.generate_report(SimpleNamespace(srcdir=tmp_path))  # no runs at all is fine too
    assert "No published release report is available yet" in (tmp_path / "report" / "index.rst").read_text()


def test_a_run_yaml_needs_a_title_and_summary(tmp_path):
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "run.yaml").write_text("title: Only a title\n")
    with pytest.raises(ValueError, match="summary"):
        report_gen.discover_runs(tmp_path)
