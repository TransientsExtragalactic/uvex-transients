"""Tests for `uvex_transients.utils.updates`, the outdated-version warning."""

import json
import time

import pytest
from packaging.version import Version

from uvex_transients.utils import updates


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """Enable the check against an empty temporary cache, with no real network and a fixed installed version."""
    monkeypatch.delenv(updates.DISABLE_ENV_VAR, raising=False)
    monkeypatch.setattr(updates, "cache_dir", tmp_path)
    monkeypatch.setattr(updates, "_installed_version", lambda: Version("0.1.0a0"))

    def no_network(timeout):
        raise AssertionError("unexpected network access")

    monkeypatch.setattr(updates, "_fetch_latest_tag", no_network)
    return tmp_path


def _write_cache(path, latest, age_hours=0.0):
    (path / "update_check.json").write_text(
        json.dumps({"checked_at": time.time() - age_hours * 3600, "latest": latest})
    )


def test_warns_when_cached_release_is_newer(isolated):
    _write_cache(isolated, "v0.2.0alpha")
    with pytest.warns(updates.OutdatedVersionWarning, match="v0.2.0alpha"):
        assert updates.check_for_updates() == "v0.2.0alpha"


@pytest.mark.parametrize("latest", ["v0.1.0alpha", "v0.0.7alpha", "not-a-version", None])
def test_silent_when_up_to_date_or_unparsable(isolated, recwarn, latest):
    _write_cache(isolated, latest)
    assert updates.check_for_updates() is None
    assert not recwarn.list


def test_silent_for_development_builds(isolated, monkeypatch, recwarn):
    monkeypatch.setattr(updates, "_installed_version", lambda: Version("0.0.7a1.dev36"))
    _write_cache(isolated, "v0.2.0alpha")
    assert updates.check_for_updates() is None
    assert not recwarn.list


def test_disabled_by_environment_variable(isolated, monkeypatch, recwarn):
    monkeypatch.setenv(updates.DISABLE_ENV_VAR, "1")
    _write_cache(isolated, "v0.2.0alpha", age_hours=1000)
    assert updates.check_for_updates(background=False) is None
    assert not recwarn.list


def test_fresh_cache_is_not_refreshed(isolated):
    """A cache younger than the interval answers without touching the network (the fixture forbids it)."""
    _write_cache(isolated, "v0.1.0alpha", age_hours=1)
    assert updates.check_for_updates(background=False) is None


def test_stale_cache_is_refreshed_and_reported(isolated, monkeypatch):
    monkeypatch.setattr(updates, "_fetch_latest_tag", lambda timeout: "v0.3.0alpha")
    _write_cache(isolated, "v0.1.0alpha", age_hours=48)
    with pytest.warns(updates.OutdatedVersionWarning, match="v0.3.0alpha"):
        assert updates.check_for_updates(background=False) == "v0.3.0alpha"
    assert json.loads((isolated / "update_check.json").read_text())["latest"] == "v0.3.0alpha"


def test_empty_cache_is_populated_in_the_background(isolated, monkeypatch, recwarn):
    """With no cache the first call can't warn; it only starts the refresh."""
    monkeypatch.setattr(updates, "_fetch_latest_tag", lambda timeout: "v0.3.0alpha")
    assert updates.check_for_updates(background=False) == "v0.3.0alpha"  # synchronous refresh path
    recwarn.clear()
    assert updates.check_for_updates(background=True) == "v0.3.0alpha"  # now served from the cache


def test_failed_refresh_never_raises_and_records_the_attempt(isolated, monkeypatch, recwarn):
    def offline(timeout):
        raise OSError("offline")

    monkeypatch.setattr(updates, "_fetch_latest_tag", offline)
    assert updates.check_for_updates(background=False) is None
    assert not recwarn.list
    assert json.loads((isolated / "update_check.json").read_text())["latest"] is None

    # The failed attempt is cached, so the next call does not retry the network.
    monkeypatch.setattr(updates, "_fetch_latest_tag", lambda timeout: pytest.fail("retried"))
    assert updates.check_for_updates(background=False) is None


def test_failed_refresh_keeps_the_previous_answer(isolated, monkeypatch):
    def offline(timeout):
        raise OSError("offline")

    monkeypatch.setattr(updates, "_fetch_latest_tag", offline)
    _write_cache(isolated, "v0.2.0alpha", age_hours=48)
    with pytest.warns(updates.OutdatedVersionWarning):
        assert updates.check_for_updates(background=False) == "v0.2.0alpha"
