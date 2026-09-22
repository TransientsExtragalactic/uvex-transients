"""Tests for `uvex_transients.surveys.utils` schedule-registry helpers."""

import pytest
import requests

from uvex_transients.surveys.utils import list_schedules
from uvex_transients.utils import config


@pytest.mark.network
@pytest.mark.parametrize("name", list_schedules())
def test_schedule_url_resolves(name):
    """Every URL registered under `config["schedules.schedule_urls"]` should be reachable.

    This only checks that the URL resolves (HTTP 200, following redirects) -- it does not
    download or validate the schedule table itself, so it stays cheap even though it hits
    the network.
    """
    url = config["schedules.schedule_urls"][name]
    response = requests.head(url, allow_redirects=True, timeout=30)

    assert response.status_code == 200, f"Schedule {name!r} at {url} returned HTTP {response.status_code}."


def test_default_schedule_is_registered():
    """`schedules.default_schedule` must itself be one of `schedules.schedule_urls`'s keys."""
    assert config["schedules.default_schedule"] in list_schedules()
