"""Tests for `uvex_transients.utils.filter_utils`, using a local VOTable in place of the SVO server."""

from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest
from astropy import units as u
from astropy.io.votable import from_table
from astropy.table import Table

from uvex_transients.utils import filter_utils


@pytest.fixture
def fake_svo(tmp_path, monkeypatch):
    """Replace `download_file` with one that serves a fixed VOTable and records its calls."""
    table = Table({"Wavelength": [2000.0, 2100.0, 2200.0] * u.AA, "Transmission": [0.0, 0.5, 0.1]})
    path = tmp_path / "response.xml"
    from_table(table).to_xml(str(path))
    calls = []

    def fake_download(url, **kwargs):
        calls.append((url, kwargs))
        return str(path)

    monkeypatch.setattr(filter_utils, "download_file", fake_download)
    return calls


def test_get_transmission_data(fake_svo):
    table = filter_utils.get_transmission_data("Swift/UVOT.UVW1")
    assert table["Wavelength"].unit == u.AA
    assert np.allclose(table["Transmission"], [0.0, 0.5, 0.1])

    url, kwargs = fake_svo[0]
    assert url.startswith(filter_utils.SVO_URL)
    assert parse_qs(urlparse(url).query) == {"ID": ["Swift/UVOT.UVW1"]}
    assert kwargs["cache"] is True


def test_get_filter_list_drops_none_instrument(fake_svo):
    filter_utils.get_filter_list("Swift")
    assert parse_qs(urlparse(fake_svo[0][0]).query) == {"Facility": ["Swift"]}

    filter_utils.get_filter_list("Swift", instrument="UVOT", cache="update")
    url, kwargs = fake_svo[1]
    assert parse_qs(urlparse(url).query) == {"Facility": ["Swift"], "Instrument": ["UVOT"]}
    assert kwargs["cache"] == "update"


def test_invalid_parameter_raises_before_any_request(fake_svo):
    with pytest.raises(ValueError, match="Bogus"):
        filter_utils.query_svo({"Bogus": 1})
    assert fake_svo == []


def test_empty_response_raises_index_error(tmp_path, monkeypatch):
    path = tmp_path / "empty.xml"
    path.write_text('<?xml version="1.0"?><VOTABLE version="1.4" xmlns="http://www.ivoa.net/xml/VOTable/v1.3"/>')
    monkeypatch.setattr(filter_utils, "download_file", lambda url, **kw: str(path))
    with pytest.raises(IndexError, match="nothing here"):
        filter_utils.query_svo({"ID": "x"}, error_msg="nothing here")


@pytest.mark.network
def test_swift_uvot_filters_from_svo():
    """The real service lists the three UVOT UV filters and returns a transmission curve."""
    ids = list(filter_utils.get_filter_list("Swift", "UVOT")["filterID"])
    assert {"Swift/UVOT.UVW1", "Swift/UVOT.UVM2", "Swift/UVOT.UVW2"} <= set(ids)
    assert len(filter_utils.get_transmission_data("Swift/UVOT.UVW1")) > 10
