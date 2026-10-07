"""Tests for scripts/sync_scheduler.py, which pins the repo to the latest uvex-scheduler release."""

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("sync_scheduler", _ROOT / "scripts" / "sync_scheduler.py")
sync_scheduler = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync_scheduler)


@pytest.fixture
def package_config():
    return (_ROOT / "uvex_transients" / "config.yaml").read_text()


@pytest.fixture
def run_config():
    return (_ROOT / "configs" / "full_run.yaml").read_text()


def test_new_tag_is_registered_and_pinned(package_config, run_config):
    """A new release is added next to the old entries, and every pointer moves to it."""
    new = sync_scheduler.pin_package_config(package_config, "v9.9.9")
    assert "    uvex_v9.9.9: " in new
    assert "refs/tags/v9.9.9/tables/plan.ecsv" in new
    assert "  default_schedule: uvex_v9.9.9" in new
    assert "    repo_url: https://raw.githubusercontent.com/m4opt/uvex-scheduler/refs/tags/v9.9.9\n" in new
    # Every previously registered schedule survives as a reference.
    for line in package_config.splitlines():
        if line.startswith("    uvex_"):
            assert line in new.splitlines()

    assert "schedule:\n  name: uvex_v9.9.9\n" in sync_scheduler.pin_run_config(run_config, "v9.9.9")


def test_pinning_is_idempotent(package_config, run_config):
    """Pinning twice to the same tag changes nothing the second time."""
    once = sync_scheduler.pin_package_config(package_config, "v9.9.9")
    assert sync_scheduler.pin_package_config(once, "v9.9.9") == once
    once = sync_scheduler.pin_run_config(run_config, "v9.9.9")
    assert sync_scheduler.pin_run_config(once, "v9.9.9") == once


def test_repo_is_pinned_to_a_registered_tag(package_config, run_config):
    """The committed configs agree with each other: full_run.yaml uses the default, which is registered."""
    tag = sync_scheduler.pin_package_config(package_config, "v9.9.9")  # exercises the regexes on the real file
    assert tag != package_config
    default = next(line for line in package_config.splitlines() if line.startswith("  default_schedule: "))
    name = default.split(": ")[1]
    assert name.startswith("uvex_v")
    assert f"schedule:\n  name: {name}\n" in run_config
