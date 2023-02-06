from unittest.mock import Mock

import pytest
import requests
from minimalkv.memory import DictStore

from startupradar.transformers.util.api import (
    StartupRadarAPI,
    MinimalKeyValueCache,
    CachedResponse,
    ResponseStatus,
)
from startupradar.transformers.util.exceptions import NotFoundError


@pytest.fixture()
def api():
    # use a dictionary as cache
    cache = MinimalKeyValueCache(DictStore())

    # don't set api key to ensure requests fail
    api = StartupRadarAPI(None, cache=cache)

    return api


def test_cache_works_without_requests(api):
    api.cache.put("/web/domains/karllorey.com", CachedResponse(ResponseStatus.OK))
    assert api.get_domain("karllorey.com") is None


def test_cache_throws_not_found(api):
    api.cache.put(
        "/web/domains/karllorey.com", CachedResponse(ResponseStatus.NOT_FOUND)
    )
    with pytest.raises(NotFoundError):
        api.get_domain("karllorey.com")


def test_cached_response_conversion():
    data = {"bla": 1, "test": False}
    resp = CachedResponse(ResponseStatus.OK, data)
    resp_byes = resp.to_bytes()
    resp_from_bytes = CachedResponse.from_bytes(resp_byes)
    assert resp_from_bytes.data == data


def test_api_detects_if_cached():
    api = StartupRadarAPI("demo", session_factory=requests.Session, cache=None)
    assert api.is_cached is False


def test_api_404_gets_cached():
    mock_response = Mock()
    mock_response.status_code = 404

    mock_session = Mock()
    mock_session.get = Mock(return_value=mock_response)

    mock_session_factory = lambda: mock_session
    cache = MinimalKeyValueCache(store=DictStore())
    api = StartupRadarAPI(
        "wrong",
        session_factory=mock_session_factory,
        cache=cache,
    )

    for _ in range(2):
        with pytest.raises(NotFoundError):
            api.get_whois("blabla.de")

    assert mock_session.get.call_count == 1
