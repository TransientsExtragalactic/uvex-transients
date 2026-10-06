"""Tests for `uvex_transients.utils.results.get_results`, with the GitHub API and downloads mocked out."""

import importlib

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
