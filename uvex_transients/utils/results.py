"""
Downloading the published simulation results attached to UVEX Transients GitHub releases.

Every tagged release runs ``configs/full_run.yaml`` and attaches the outputs to the release as
assets. Secondary simulations (for example ``configs/infant_sne.yaml``) are attached too, each as one
``<simulation>.zip`` holding the same four files. They are not part of the pip-installed package, so
`get_results` fetches them on demand, keeps them in the package cache, and returns local paths for
the usual ``from_disk`` loaders. Only the requested files are downloaded from a simulation's archive.
"""

import io
import json
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

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
_RANGE_BLOCK = 1 << 20
_ARCHIVE_SUFFIX = ".zip"


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
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context), _SameHostAuthRedirect())
    return opener.open(request, timeout=timeout)


class _SameHostAuthRedirect(urllib.request.HTTPRedirectHandler):
    """Follow redirects, but drop the ``Authorization`` header when they leave the original host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """
        Build the redirected request, without credentials if it goes to a different host.

        Parameters
        ----------
        req : urllib.request.Request
            The original request.
        fp : file-like
            The response body of the redirect.
        code : int
            The redirect's HTTP status code.
        msg : str
            The redirect's HTTP status message.
        headers : http.client.HTTPMessage
            The redirect's response headers.
        newurl : str
            The URL being redirected to.

        Returns
        -------
        urllib.request.Request or None
            The request to follow, or `None` if the redirect should not be followed.
        """
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlparse(newurl).netloc != urllib.parse.urlparse(req.full_url).netloc:
            new.remove_header("Authorization")
        return new


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
    the ``/releases/latest`` endpoint, since the latter excludes prereleases, which some UVEX
    Transients releases are.

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
    logger.info("Querying %s for the release list.", _API_URL)
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
        with (
            _urlopen(_request(url), timeout) as response,
            open(partial, "wb") as f,
            logging_redirect_tqdm(loggers=[logger]),
            tqdm(
                total=int(response.headers.get("Content-Length") or 0) or None,
                desc=f"Downloading {dest.name}",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar,
        ):
            while chunk := response.read(_CHUNK_SIZE):
                f.write(chunk)
                pbar.update(len(chunk))
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)


class _RangeUnsupported(Exception):
    """The server answered a ``Range`` request with a full body, so partial reads are not possible."""


class _RangeFile(io.RawIOBase):
    """
    A read-only, seekable file over a remote URL, fetched with HTTP ``Range`` requests.

    Lets `zipfile.ZipFile` read a remote archive's central directory and individual members
    without downloading the rest of it. Wrap in an `io.BufferedReader` so that the many small
    reads `zipfile` makes become a few large requests.

    Parameters
    ----------
    url : str
        The file's download URL. Redirects are followed on every request, so the short-lived signed
        URL that release assets redirect to never needs to be cached.
    size : int
        The file's size in bytes (known from the release metadata).
    timeout : float
        Network timeout in seconds, per request.
    """

    def __init__(self, url: str, size: int, timeout: float):
        super().__init__()
        self._url = url
        self._size = size
        self._timeout = timeout
        self._pos = 0

    def readable(self) -> bool:
        """
        Report that the file can be read.

        Returns
        -------
        bool
            Always `True`.
        """
        return True

    def seekable(self) -> bool:
        """
        Report that the file supports random access.

        Returns
        -------
        bool
            Always `True`.
        """
        return True

    def tell(self) -> int:
        """
        Return the current read position.

        Returns
        -------
        int
            The offset, in bytes from the start of the file.
        """
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """
        Move the read position. No request is made until the next read.

        Parameters
        ----------
        offset : int
            The offset in bytes, relative to `whence`.
        whence : int, optional
            `io.SEEK_SET` (from the start of the file, the default), `io.SEEK_CUR` (from the current
            position) or `io.SEEK_END` (from the end of the file).

        Returns
        -------
        int
            The new position, in bytes from the start of the file.

        Raises
        ------
        ValueError
            If `whence` is not one of the three above, or the new position is negative.
        """
        if whence == io.SEEK_SET:
            pos = offset
        elif whence == io.SEEK_CUR:
            pos = self._pos + offset
        elif whence == io.SEEK_END:
            pos = self._size + offset
        else:
            raise ValueError(f"invalid whence ({whence})")
        if pos < 0:
            raise ValueError("negative seek position")
        self._pos = pos
        return pos

    def readinto(self, buffer) -> int:
        """
        Read bytes at the current position into `buffer`, with one ``Range`` request.

        Parameters
        ----------
        buffer : bytearray or memoryview
            Writable buffer to fill. At most ``len(buffer)`` bytes are requested, fewer near the end of the file.

        Returns
        -------
        int
            The number of bytes read, 0 at the end of the file.

        Raises
        ------
        _RangeUnsupported
            If the server answers with the whole file instead of the requested range.
        """
        view = memoryview(buffer).cast("B")
        end = min(self._pos + len(view), self._size)
        if end <= self._pos:
            return 0
        request = _request(self._url)
        request.add_header("Range", f"bytes={self._pos}-{end - 1}")
        with _urlopen(request, self._timeout) as response:
            if response.status != 206:
                raise _RangeUnsupported(f"{self._url} ignored a Range request (HTTP {response.status}).")
            data = response.read()
        view[: len(data)] = data
        self._pos += len(data)
        return len(data)


def _archive_name(simulation: str) -> str:
    """Return the release asset name of `simulation`'s archive."""
    return f"{simulation}{_ARCHIVE_SUFFIX}"


def _simulation_names(release: dict) -> list[str]:
    """Return the sorted simulation names among `release`'s ``<simulation>.zip`` assets."""
    return sorted(
        asset["name"][: -len(_ARCHIVE_SUFFIX)]
        for asset in release.get("assets", [])
        if asset["name"].endswith(_ARCHIVE_SUFFIX)
    )


def list_simulations(tag: str | None = None, timeout: float = 60.0) -> list[str]:
    """
    List the secondary simulations attached to a published release.

    Each is a ``<simulation>.zip`` asset; pass the name to `get_results` as ``simulation``.

    Parameters
    ----------
    tag : str, optional
        The release tag. The default is the newest published release.
    timeout : float, optional
        Network timeout in seconds. The default is 60.

    Returns
    -------
    list of str
        The simulation names, sorted.

    Raises
    ------
    ConnectionError
        If the GitHub releases API cannot be reached.
    LookupError
        If there are no published releases, or none carries `tag`.
    """
    return _simulation_names(find_release(tag, timeout))


def _extract_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, dest: Path) -> None:
    """Stream one archive member to `dest` through a ``.part`` file, so a failure never leaves a truncated `dest`."""
    partial = dest.with_name(dest.name + ".part")
    try:
        with (
            archive.open(info) as source,
            open(partial, "wb") as f,
            logging_redirect_tqdm(loggers=[logger]),
            tqdm(
                total=info.file_size or None,
                desc=f"Extracting {dest.name}",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar,
        ):
            while chunk := source.read(_CHUNK_SIZE):
                f.write(chunk)
                pbar.update(len(chunk))
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)


def _get_simulation_results(
    release: dict,
    simulation: str,
    assets: list[str] | None,
    out_dir: Path,
    overwrite: bool,
    timeout: float,
) -> dict[str, Path]:
    """
    Fetch the requested files of one simulation's archive into `out_dir`, downloading only those members.

    See `get_results`, which resolves `release` and `out_dir` and documents the other arguments.
    """
    tag = release["tag_name"]
    archive_name = _archive_name(simulation)
    available = {asset["name"]: asset for asset in release.get("assets", [])}
    if archive_name not in available:
        raise LookupError(
            f"Release {tag} has no simulation {simulation!r} (no asset {archive_name!r}). "
            f"Available simulations: {_simulation_names(release)}."
        )
    asset = available[archive_name]

    remote = io.BufferedReader(_RangeFile(asset["browser_download_url"], asset["size"], timeout), _RANGE_BLOCK)
    fallback = None
    try:
        try:
            archive = zipfile.ZipFile(remote)
        except _RangeUnsupported:
            logger.info("The server does not support Range requests; downloading all of %s.", archive_name)
            fallback = out_dir / (archive_name + ".download")
            _download(asset["browser_download_url"], fallback, timeout)
            archive = zipfile.ZipFile(fallback)

        with archive:
            members = {info.filename: info for info in archive.infolist() if not info.is_dir()}
            requested = list(members) if assets is None else list(assets)
            filenames = {name: RESULT_ASSETS.get(name, name) for name in requested}
            missing = sorted({filename for filename in filenames.values() if filename not in members})
            if missing:
                raise LookupError(
                    f"Simulation {simulation!r} of release {tag} has no file(s) {missing}. "
                    f"Available: {sorted(members)}."
                )

            paths = {}
            for name, filename in filenames.items():
                info = members[filename]
                dest = out_dir / filename
                if dest.exists():
                    if dest.stat().st_size == info.file_size:
                        logger.info("Results file %s already present at %s.", filename, dest)
                        paths[name] = dest
                        continue
                    if not overwrite:
                        raise FileExistsError(
                            f"{dest} differs in size from {archive_name}'s {filename} in release {tag}; "
                            "pass overwrite=True to replace it."
                        )
                logger.info("Extracting %s from %s of release %s to %s.", filename, archive_name, tag, dest)
                _extract_member(archive, info, dest)
                paths[name] = dest
    finally:
        remote.close()
        if fallback is not None:
            fallback.unlink(missing_ok=True)
    return paths


def get_results(
    assets: list[str] | None = None,
    tag: str | None = None,
    directory: str | os.PathLike | None = None,
    overwrite: bool = False,
    timeout: float = 60.0,
    simulation: str | None = None,
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
        ``"photometry"``, ``"summary"``) or a file name. The default is every file of the
        release (or of the `simulation`), including the large photometry table, so asking for only
        what you need is usually better.
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
    simulation : str, optional
        Fetch from a secondary simulation (see `list_simulations`, for example ``"infant_sne"``)
        instead of the release's primary run. Each is attached as ``<simulation>.zip`` holding the
        same files as the primary run, under the same `RESULT_ASSETS` keys. Only the requested
        files are downloaded from the archive, using HTTP range requests. Files are cached in
        ``<cache_dir>/results/<tag>/<simulation>/``. The default is the primary run.

    Returns
    -------
    dict of str to pathlib.Path
        Each requested name, as given in `assets`, mapped to its local path.

    Raises
    ------
    ConnectionError
        If the GitHub releases API cannot be reached.
    LookupError
        If there are no published releases, `tag` does not exist, `simulation` is not on the
        release, or a requested asset is not on the release (or in the simulation).
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

        # The same files from a secondary simulation.
        infant = get_results(
            ["summary"], simulation="infant_sne"
        )
    """
    release = find_release(tag, timeout)
    tag = release["tag_name"]

    if simulation is not None:
        default_dir = cache_dir / "results" / tag / simulation
        out_dir = Path(directory).expanduser().resolve() if directory is not None else default_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        return _get_simulation_results(release, simulation, assets, out_dir, overwrite, timeout)

    available = {asset["name"]: asset for asset in release.get("assets", [])}

    requested = [name for name in available if not name.endswith(_ARCHIVE_SUFFIX)] if assets is None else list(assets)
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
        logger.info("Downloaded %s (%d bytes).", filename, dest.stat().st_size)
        paths[name] = dest

    return paths
