"""
Downloading the published simulation results attached to UVEX Transients GitHub releases.

Every tagged release runs ``configs/full_run.yaml`` and attaches the outputs to the release as
assets. They are not part of the pip-installed package, so `get_results` fetches them on demand,
keeps them in the package cache, and returns local paths for the usual ``from_disk`` loaders.
"""

import json
import os
import ssl
import urllib.error
import urllib.request
from pathlib import Path

from .config import cache_dir
from .log import logger

GITHUB_REPO = "TransientsExtragalactic/uvex-transients"
"""str: The ``owner/name`` GitHub repository whose releases hold the published results."""

RELEASES_URL = f"https://github.com/{GITHUB_REPO}/releases"
"""str: The repository's releases page."""

RESULT_ASSETS = {
    "events": "final_catalog.ecsv",
    "exposure": "exposure.ecsv",
    "photometry": "photometry.ecsv",
    "summary": "event_summary.ecsv",
}
"""dict of str to str: Short names for the release assets, mapped to their file names.

``events`` is the `~uvex_transients.simulation.event_catalog.EventCatalog` that survived the cuts,
``exposure`` the `~uvex_transients.simulation.exposure_catalog.ExposureCatalog`, ``photometry`` the
synthetic photometry of those events (the largest file, over 100 MB), and ``summary`` the per-event
summary table from `~uvex_transients.simulation.core.SurveySimulator.run_event_summary_action`.
"""

_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases"
_CHUNK_SIZE = 1 << 20


def _urlopen(request: urllib.request.Request, timeout: float):
    """
    Open `request`, verifying TLS against ``certifi``'s CA bundle when it is installed.

    Some Python builds (notably the python.org macOS installers) ship without system CA
    certificates, which makes a plain `urllib.request.urlopen` fail verification on GitHub.

    Parameters
    ----------
    request : urllib.request.Request
        The request to open.
    timeout : float
        Network timeout in seconds.

    Returns
    -------
    http.client.HTTPResponse
        The open response, usable as a context manager.
    """
    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = None
    return urllib.request.urlopen(request, timeout=timeout, context=context)


def _request(url: str) -> urllib.request.Request:
    """Build a GitHub request for `url`, authenticated if ``$GITHUB_TOKEN`` is set (to avoid API rate limits)."""
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


def find_release(tag: str | None = None, timeout: float = 60.0) -> dict:
    """
    Look up one published GitHub release.

    With no `tag`, the newest published release is used. The releases list is read rather than
    the ``/releases/latest`` endpoint, since the latter excludes prereleases and every UVEX
    Transients release so far is one.

    Parameters
    ----------
    tag : str, optional
        The release tag (for example ``"v0.1.1alpha"``). The default is the newest release.
    timeout : float, optional
        Network timeout in seconds. The default is 60.

    Returns
    -------
    dict
        The release's GitHub metadata, including ``tag_name``, ``html_url`` and ``assets``.

    Raises
    ------
    ConnectionError
        If the GitHub releases API cannot be reached.
    LookupError
        If there are no published releases, or none carries `tag`.
    """
    try:
        with _urlopen(_request(_API_URL), timeout) as response:
            releases = json.load(response)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as err:
        raise ConnectionError(f"Could not reach the GitHub releases API at {_API_URL}: {err}") from err

    published = [release for release in releases if not release.get("draft", False)]
    if tag is None:
        if not published:
            raise LookupError(f"No published releases found for {GITHUB_REPO}.")
        return published[0]

    for release in published:
        if release.get("tag_name") == tag:
            return release
    raise LookupError(f"No release tagged {tag!r} for {GITHUB_REPO}. Available: {[r['tag_name'] for r in published]}.")


def _download(url: str, dest: Path, timeout: float) -> None:
    """
    Stream `url` to `dest` through a ``.part`` file, so a failed download never leaves a truncated `dest`.

    Parameters
    ----------
    url : str
        The asset's download URL.
    dest : pathlib.Path
        Final destination path.
    timeout : float
        Network timeout in seconds.
    """
    partial = dest.with_name(dest.name + ".part")
    try:
        with _urlopen(_request(url), timeout) as response, open(partial, "wb") as f:
            while chunk := response.read(_CHUNK_SIZE):
                f.write(chunk)
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)


def get_results(
    assets: list[str] | None = None,
    tag: str | None = None,
    directory: str | os.PathLike | None = None,
    overwrite: bool = False,
    timeout: float = 60.0,
) -> dict[str, Path]:
    """
    Download assets of a published release and return their local paths.

    Files already present with the expected size are not downloaded again, so repeated calls only
    cost one small request to the GitHub API, to learn which release is newest. By default the
    files are kept in the package cache directory, in a subdirectory per release
    (``<cache_dir>/results/<tag>/``), so different releases never mix.

    Parameters
    ----------
    assets : list of str, optional
        What to fetch, each either a key of `RESULT_ASSETS` (``"events"``, ``"exposure"``,
        ``"photometry"``, ``"summary"``) or an asset file name. The default is every asset of the
        release, including the large photometry table, so asking for only what you need is
        usually better.
    tag : str, optional
        The release tag to fetch. The default is the newest published release.
    directory : str or os.PathLike, optional
        Place the files directly in this directory instead of the per-release cache directory.
    overwrite : bool, optional
        Replace an existing file whose size differs from the release asset. The default raises
        instead, so an explicit `directory` holding results from another release is never
        silently overwritten.
    timeout : float, optional
        Network timeout in seconds, per request. The default is 60.

    Returns
    -------
    dict of str to pathlib.Path
        Each requested name, as given in `assets`, mapped to its local path.

    Raises
    ------
    ConnectionError
        If the GitHub releases API cannot be reached.
    LookupError
        If there are no published releases, `tag` does not exist, or a requested asset is not on
        the release.
    FileExistsError
        If a file already exists with a different size than the release asset and `overwrite` is
        False.

    Examples
    --------
    .. code-block:: python

        from uvex_transients.simulation import (
            EventCatalog,
        )
        from uvex_transients.utils import get_results

        paths = get_results(["events", "summary"])
        events = EventCatalog.from_disk(paths["events"])
    """
    release = find_release(tag, timeout)
    tag = release["tag_name"]
    available = {asset["name"]: asset for asset in release.get("assets", [])}

    requested = list(available) if assets is None else list(assets)
    filenames = {name: RESULT_ASSETS.get(name, name) for name in requested}
    missing = sorted({filename for filename in filenames.values() if filename not in available})
    if missing:
        raise LookupError(f"Release {tag} has no asset(s) {missing}. Available: {sorted(available)}.")

    out_dir = Path(directory).expanduser().resolve() if directory is not None else cache_dir / "results" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for name, filename in filenames.items():
        asset = available[filename]
        dest = out_dir / filename
        if dest.exists():
            if dest.stat().st_size == asset["size"]:
                logger.info("Results asset %s already present at %s.", filename, dest)
                paths[name] = dest
                continue
            if not overwrite:
                raise FileExistsError(
                    f"{dest} differs in size from release {tag}'s {filename}; pass overwrite=True to replace it."
                )
        logger.info("Downloading %s from release %s to %s.", filename, tag, dest)
        _download(asset["browser_download_url"], dest, timeout)
        paths[name] = dest

    return paths
