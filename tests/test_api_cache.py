import pytest
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
