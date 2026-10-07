"""
Warn when the installed UVEX Transients is older than the latest GitHub release.

The check never delays or breaks an import. Importing the package only *reads* a small cache
file (``update_check.json`` in the package cache directory) and warns if that file records a newer
release than the one installed. When the cache is older than ``system.update_check.interval_hours``,
a background thread refreshes it from the GitHub releases API, so a newly published release is
reported the next time the package is imported. Any network or file problem is swallowed.

Turn the check off with ``system.update_check.enabled: false`` in the config, or by setting the
``UVEX_TRANSIENTS_NO_UPDATE_CHECK`` environment variable. Development builds (versions containing
``.dev``) are never checked.
"""

import json
import os
import threading
import time
import warnings

from packaging.version import InvalidVersion, Version

from .config import cache_dir, config
from .log import logger
from .results import _API_URL, GITHUB_REPO, RELEASES_URL, _request, _urlopen

DISABLE_ENV_VAR = "UVEX_TRANSIENTS_NO_UPDATE_CHECK"
"""str: Setting this environment variable (to anything) turns the update check off."""

_CACHE_FILE = "update_check.json"


class OutdatedVersionWarning(UserWarning):
    """Warning issued when a newer UVEX Transients release than the installed one exists."""


def _setting(key: str, default):
    """Read ``system.update_check.<key>`` from the config, tolerating a user config that predates it."""
    try:
        return config[f"system.update_check.{key}"]
    except KeyError:
        return default


def _installed_version() -> Version | None:
    """
    Return the installed package version, or `None` if it can't be determined.

    Returns
    -------
    packaging.version.Version or None
        `None` if the generated ``_version.py`` is missing (an uninstalled source tree).
    """
    try:
        from .._version import __version__

        return Version(__version__)
    except (ImportError, InvalidVersion):
        return None


def _parse_tag(tag: str) -> Version | None:
    """Parse a release tag such as ``"v0.2.0alpha"``, or return `None` if it isn't a version."""
    try:
        return Version(tag.removeprefix("v"))
    except InvalidVersion:
        return None


def _read_cache() -> dict:
    """Read the cached check result, or an empty dict if there is none or it is unreadable."""
    try:
        data = json.loads((cache_dir / _CACHE_FILE).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_cache(data: dict) -> None:
    """Write the cached check result, ignoring a read-only or missing cache directory."""
    try:
        (cache_dir / _CACHE_FILE).write_text(json.dumps(data))
    except OSError as err:
        logger.debug("Could not write the update-check cache: %s", err)


def _fetch_latest_tag(timeout: float) -> str | None:
    """
    Ask the GitHub releases API for the tag of the newest published release.

    Parameters
    ----------
    timeout : float
        Network timeout in seconds.

    Returns
    -------
    str or None
        The newest non-draft release's tag, or `None` if there are no releases.
    """
    with _urlopen(_request(f"{_API_URL}?per_page=5"), timeout) as response:
        releases = json.load(response)
    for release in releases:
        if not release.get("draft", False):
            return release["tag_name"]
    return None


def _refresh_cache(timeout: float) -> None:
    """
    Query GitHub and record the result, keeping the previous answer if the query fails.

    The time of the attempt is stored either way, so an offline machine retries once per interval
    instead of on every import.
    """
    latest = _read_cache().get("latest")
    try:
        latest = _fetch_latest_tag(timeout) or latest
    except (OSError, ValueError, KeyError, TypeError) as err:
        logger.debug("Update check against %s failed: %s", GITHUB_REPO, err)
    _write_cache({"checked_at": time.time(), "latest": latest})


def check_for_updates(background: bool = True) -> str | None:
    """
    Warn if the cached latest release is newer than the installed version, refreshing a stale cache.

    Parameters
    ----------
    background : bool, optional
        If `True` (the default), a stale cache is refreshed in a daemon thread and the result is
        only reported on a later call. If `False`, the refresh is done first, in this thread, so a
        newer release is reported by this very call.

    Returns
    -------
    str or None
        The tag of the newer release a warning was issued for, or `None` if the check is
        disabled, the installed version is a development build, or it is up to date.
    """
    if os.environ.get(DISABLE_ENV_VAR) or not _setting("enabled", True):
        return None

    installed = _installed_version()
    if installed is None or installed.is_devrelease:
        return None

    cached = _read_cache()
    interval = float(_setting("interval_hours", 24.0)) * 3600.0
    if time.time() - float(cached.get("checked_at", 0.0)) > interval:
        timeout = float(_setting("timeout", 3.0))
        if background:
            threading.Thread(target=_refresh_cache, args=(timeout,), daemon=True, name="uvex-update-check").start()
        else:
            _refresh_cache(timeout)
            cached = _read_cache()

    latest = cached.get("latest")
    parsed = _parse_tag(latest) if isinstance(latest, str) else None
    if parsed is None or parsed <= installed:
        return None

    warnings.warn(
        f"uvex_transients {installed} is installed, but {latest} is the latest release ({RELEASES_URL}/tag/{latest}). "
        f"Upgrade with `pip install --upgrade --pre uvex-transients`. Newer releases can change the "
        f"simulated catalogs. Silence this check by setting ${DISABLE_ENV_VAR} or "
        f"`system.update_check.enabled: false` in the config.",
        OutdatedVersionWarning,
        stacklevel=2,
    )
    return latest
