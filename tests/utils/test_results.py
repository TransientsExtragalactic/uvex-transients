"""Tests for `uvex_transients.utils.results.get_results`, with the GitHub API and downloads mocked out."""

import http.server
import importlib
import io
import os
import threading
import zipfile

import pytest

results = importlib.import_module("uvex_transients.utils.results")

RELEASES = [
    {"tag_name": "v2", "draft": True, "assets": []},
    {
        "tag_name": "v1",
        "draft": False,
        "assets": [
            {"name": "event_summary.ecsv", "size": 3, "browser_download_url": "https://example.test/summary"},
            {"name": "exposure.ecsv", "size": 5, "browser_download_url": "https://example.test/exposure"},
        ],
    },
    {
        "tag_name": "v0",
        "draft": False,
        "assets": [{"name": "event_summary.ecsv", "size": 1, "browser_download_url": "https://example.test/old"}],
    },
]
PAYLOADS = {
    "https://example.test/summary": b"abc",
    "https://example.test/exposure": b"12345",
    "https://example.test/old": b"x",
}


@pytest.fixture
def fake_github(monkeypatch, tmp_path):
    """Replace the release lookup and downloader, recording each download, and redirect the cache directory."""
    downloads = []

    def fake_download(url, dest, timeout):
        downloads.append(url)
        dest.write_bytes(PAYLOADS[url])

    def fake_find_release(tag=None, timeout=60.0):
        published = [r for r in RELEASES if not r["draft"]]
        if tag is None:
            return published[0]
        for release in published:
            if release["tag_name"] == tag:
                return release
        raise LookupError(tag)

    monkeypatch.setattr(results, "_download", fake_download)
    monkeypatch.setattr(results, "find_release", fake_find_release)
    monkeypatch.setattr(results, "cache_dir", tmp_path / "cache")
    return downloads


class TestGetResults:
    def test_newest_published_release_into_the_cache(self, fake_github, tmp_path):
        paths = results.get_results()
        assert set(paths) == {"event_summary.ecsv", "exposure.ecsv"}  # the draft v2 is skipped, so v1 is newest
        assert paths["event_summary.ecsv"] == tmp_path / "cache" / "results" / "v1" / "event_summary.ecsv"
        assert paths["exposure.ecsv"].read_bytes() == b"12345"

    def test_short_keys_resolve_to_file_names_and_are_returned_as_given(self, fake_github):
        paths = results.get_results(["summary"])
        assert list(paths) == ["summary"]
        assert paths["summary"].name == "event_summary.ecsv"
        assert fake_github == ["https://example.test/summary"]

    def test_a_file_name_is_accepted_directly(self, fake_github):
        paths = results.get_results(["exposure.ecsv"])
        assert list(paths) == ["exposure.ecsv"]

    def test_explicit_directory(self, fake_github, tmp_path):
        paths = results.get_results(["summary"], directory=tmp_path / "mine")
        assert paths["summary"] == (tmp_path / "mine" / "event_summary.ecsv").resolve()

    def test_explicit_tag(self, fake_github):
        paths = results.get_results(["summary"], tag="v0")
        assert paths["summary"].read_bytes() == b"x"
        assert paths["summary"].parent.name == "v0"

    def test_unknown_asset(self, fake_github):
        with pytest.raises(LookupError, match="photometry.ecsv"):
            results.get_results(["photometry"])

    def test_unknown_tag(self, fake_github):
        with pytest.raises(LookupError):
            results.get_results(tag="v99")

    def test_an_existing_file_of_the_right_size_is_not_downloaded_again(self, fake_github):
        results.get_results()
        fake_github.clear()
        results.get_results()
        assert fake_github == []

    def test_size_mismatch_raises_unless_overwrite(self, fake_github, tmp_path):
        stale = tmp_path / "mine"
        stale.mkdir()
        (stale / "event_summary.ecsv").write_bytes(b"stale-and-longer")
        with pytest.raises(FileExistsError):
            results.get_results(["summary"], directory=stale)
        paths = results.get_results(["summary"], directory=stale, overwrite=True)
        assert paths["summary"].read_bytes() == b"abc"


class TestFindRelease:
    def test_unreachable_api_raises_connection_error(self, monkeypatch):
        def boom(*args, **kwargs):
            raise OSError("offline")

        monkeypatch.setattr(results, "_urlopen", boom)
        with pytest.raises(ConnectionError):
            results.find_release()

    def test_no_published_release_raises_lookup_error(self, monkeypatch):
        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self, *args):
                return b'[{"tag_name": "v1", "draft": true, "assets": []}]'

        monkeypatch.setattr(results, "_urlopen", lambda request, timeout: _Response())
        with pytest.raises(LookupError, match="No published releases"):
            results.find_release()


# ----------------------------------------- #
# Secondary simulations (zip archives)       #
# ----------------------------------------- #
SUMMARY = b"summary-of-the-infant-run"
PHOTOMETRY = os.urandom(6 << 20)  # incompressible, so the archive is dominated by this member


def _build_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("event_summary.ecsv", SUMMARY)
        archive.writestr("exposure.ecsv", b"exposure")
        archive.writestr("photometry.ecsv", PHOTOMETRY)
    return buffer.getvalue()


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        server = self.server
        header = self.headers.get("Range")
        server.requests.append(header)
        payload = server.payload
        if header and server.support_range:
            start, end = (int(x) for x in header.removeprefix("bytes=").split("-"))
            body = payload[start : end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{start + len(body) - 1}/{len(payload)}")
        else:
            body = payload
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        server.bytes_served += len(body)

    def log_message(self, *args):
        pass


def _serve(payload: bytes, name: str):
    """Serve `payload` from a local HTTP server; ``support_range`` toggles Range support."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.payload = payload
    server.support_range = True
    server.requests = []
    server.bytes_served = 0
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    server.url = f"http://127.0.0.1:{server.server_address[1]}/{name}"
    yield server
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.fixture
def serve_zip():
    """Serve the simulation archive from a local HTTP server."""
    yield from _serve(_build_zip(), "infant_sne.zip")


@pytest.fixture
def fake_simulation_release(monkeypatch, tmp_path, serve_zip):
    """Build a release carrying a flat primary asset plus ``infant_sne.zip`` served by `serve_zip`."""
    release = {
        "tag_name": "v1",
        "draft": False,
        "assets": [
            {"name": "event_summary.ecsv", "size": 3, "browser_download_url": "https://example.test/summary"},
            {"name": "infant_sne.zip", "size": len(serve_zip.payload), "browser_download_url": serve_zip.url},
        ],
    }
    monkeypatch.setattr(results, "find_release", lambda tag=None, timeout=60.0: release)
    monkeypatch.setattr(results, "cache_dir", tmp_path / "cache")
    return release


class TestSimulations:
    """`get_results(simulation=...)` and `list_simulations`, against a local server that supports Range requests."""

    def test_fetches_only_the_requested_member(self, fake_simulation_release, serve_zip, tmp_path):
        paths = results.get_results(["summary"], simulation="infant_sne")
        assert paths["summary"] == tmp_path / "cache" / "results" / "v1" / "infant_sne" / "event_summary.ecsv"
        assert paths["summary"].read_bytes() == SUMMARY
        assert not (paths["summary"].parent / "photometry.ecsv").exists()
        assert all(header is not None for header in serve_zip.requests)  # every request was a Range request
        assert serve_zip.bytes_served < len(serve_zip.payload) / 2  # the 6 MB photometry member never moved

    def test_default_returns_every_member(self, fake_simulation_release):
        paths = results.get_results(simulation="infant_sne")
        assert set(paths) == {"event_summary.ecsv", "exposure.ecsv", "photometry.ecsv"}
        assert paths["photometry.ecsv"].read_bytes() == PHOTOMETRY

    def test_short_keys_and_file_names_are_returned_as_given(self, fake_simulation_release):
        paths = results.get_results(["summary", "exposure.ecsv"], simulation="infant_sne")
        assert list(paths) == ["summary", "exposure.ecsv"]

    def test_a_cached_member_is_not_fetched_again(self, fake_simulation_release, serve_zip):
        results.get_results(["summary"], simulation="infant_sne")
        serve_zip.requests.clear()
        results.get_results(["summary"], simulation="infant_sne")
        assert len(serve_zip.requests) > 0  # the archive's directory is still read ...
        assert serve_zip.bytes_served < len(serve_zip.payload) / 2  # ... but not the member again

    def test_explicit_directory(self, fake_simulation_release, tmp_path):
        paths = results.get_results(["summary"], simulation="infant_sne", directory=tmp_path / "mine")
        assert paths["summary"] == (tmp_path / "mine" / "event_summary.ecsv").resolve()

    def test_size_mismatch_raises_unless_overwrite(self, fake_simulation_release, tmp_path):
        stale = tmp_path / "mine"
        stale.mkdir()
        (stale / "event_summary.ecsv").write_bytes(b"stale")
        with pytest.raises(FileExistsError):
            results.get_results(["summary"], simulation="infant_sne", directory=stale)
        paths = results.get_results(["summary"], simulation="infant_sne", directory=stale, overwrite=True)
        assert paths["summary"].read_bytes() == SUMMARY

    def test_unknown_simulation_lists_the_available_ones(self, fake_simulation_release):
        with pytest.raises(LookupError, match=r"infant_sne"):
            results.get_results(simulation="nope")

    def test_unknown_member(self, fake_simulation_release):
        with pytest.raises(LookupError, match="final_catalog.ecsv"):
            results.get_results(["events"], simulation="infant_sne")

    def test_falls_back_to_a_full_download_without_range_support(self, fake_simulation_release, serve_zip):
        serve_zip.support_range = False
        paths = results.get_results(["summary"], simulation="infant_sne")
        assert paths["summary"].read_bytes() == SUMMARY
        assert not list(paths["summary"].parent.glob("*.download"))  # the temporary archive is removed

    def test_primary_run_ignores_simulation_archives(self, fake_simulation_release, monkeypatch):
        monkeypatch.setattr(results, "_download", lambda url, dest, timeout: dest.write_bytes(b"abc"))
        assert set(results.get_results()) == {"event_summary.ecsv"}

    def test_list_simulations(self, fake_simulation_release):
        assert results.list_simulations() == ["infant_sne"]


# ----------------------------------------- #
# Phase-space maps (subdirectories)          #
# ----------------------------------------- #
FUV_MAP = b"fuv-map"
NUV_MAP = b"nuv-map"


def _build_phase_space_zip(extra: dict[str, bytes] | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("fuv_detection/map.ecsv", FUV_MAP)
        archive.writestr("fuv_detection/points.ecsv", b"fuv-points")
        archive.writestr("fuv_detection/overlay.ecsv", b"fuv-overlay")
        archive.writestr("fuv_detection/map.png", b"fuv-figure")
        archive.writestr("nuv_detection/map.ecsv", NUV_MAP)
        for name, data in (extra or {}).items():
            archive.writestr(name, data)
    return buffer.getvalue()


@pytest.fixture
def serve_phase_space():
    """Serve a ``phase_space.zip`` with two maps."""
    yield from _serve(_build_phase_space_zip(), "phase_space.zip")


@pytest.fixture
def fake_phase_space_release(monkeypatch, tmp_path, serve_phase_space):
    """Build a release carrying ``phase_space.zip`` served by `serve_phase_space`."""
    release = {
        "tag_name": "v1",
        "draft": False,
        "assets": [
            {
                "name": "phase_space.zip",
                "size": len(serve_phase_space.payload),
                "browser_download_url": serve_phase_space.url,
            },
        ],
    }
    monkeypatch.setattr(results, "find_release", lambda tag=None, timeout=60.0: release)
    monkeypatch.setattr(results, "cache_dir", tmp_path / "cache")
    return release


class TestPhaseSpaceMaps:
    """`get_phase_space_map`, whose files sit in one subdirectory per map of the archive."""

    def test_fetches_the_files_of_one_map(self, fake_phase_space_release, tmp_path):
        paths = results.get_phase_space_map("fuv_detection", ["map", "points"])
        assert set(paths) == {"map", "points"}
        assert paths["map"] == tmp_path / "cache" / "results" / "v1" / "phase_space" / "fuv_detection" / "map.ecsv"
        assert paths["map"].read_bytes() == FUV_MAP

    def test_the_data_products_are_the_default(self, fake_phase_space_release):
        assert set(results.get_phase_space_map("fuv_detection")) == {"map", "points", "overlay"}

    def test_the_maps_do_not_mix(self, fake_phase_space_release):
        fuv = results.get_phase_space_map("fuv_detection", ["map"])["map"]
        nuv = results.get_phase_space_map("nuv_detection", ["map"])["map"]
        assert fuv.read_bytes() == FUV_MAP and nuv.read_bytes() == NUV_MAP and fuv != nuv

    def test_a_map_that_is_not_in_the_archive(self, fake_phase_space_release):
        with pytest.raises(LookupError, match="nope/map.ecsv"):
            results.get_phase_space_map("nope", ["map"])

    def test_a_file_that_is_not_in_the_map(self, fake_phase_space_release):
        with pytest.raises(LookupError, match="nuv_detection/points.ecsv"):
            results.get_phase_space_map("nuv_detection", ["points"])

    def test_an_unknown_file_name(self, fake_phase_space_release):
        with pytest.raises(ValueError, match="bogus"):
            results.get_phase_space_map("fuv_detection", ["bogus"])

    def test_a_release_without_maps(self, monkeypatch, tmp_path):
        release = {"tag_name": "v1", "draft": False, "assets": []}
        monkeypatch.setattr(results, "find_release", lambda tag=None, timeout=60.0: release)
        monkeypatch.setattr(results, "cache_dir", tmp_path / "cache")
        with pytest.raises(LookupError, match="phase_space"):
            results.get_phase_space_map("fuv_detection")

    def test_a_member_cannot_escape_the_cache(self, monkeypatch, tmp_path):
        payload = _build_phase_space_zip({"../escaped.ecsv": b"x"})
        server = next(_serve(payload, "phase_space.zip"))
        release = {
            "tag_name": "v1",
            "draft": False,
            "assets": [{"name": "phase_space.zip", "size": len(payload), "browser_download_url": server.url}],
        }
        monkeypatch.setattr(results, "find_release", lambda tag=None, timeout=60.0: release)
        monkeypatch.setattr(results, "cache_dir", tmp_path / "cache")
        try:
            with pytest.raises(ValueError, match="outside"):
                results.get_results(["../escaped.ecsv"], simulation="phase_space")
        finally:
            server.shutdown()
            server.server_close()
        assert not (tmp_path / "cache" / "results" / "v1" / "escaped.ecsv").exists()

    def test_it_is_listed_as_a_simulation(self, fake_phase_space_release):
        assert results.list_simulations() == ["phase_space"]
