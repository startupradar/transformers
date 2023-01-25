from unittest.mock import Mock

import pytest
import requests
from minimalkv.memory import DictStore

from startupradar.transformers.util.api import StartupRadarAPI, MinimalKeyValueCache
from startupradar.transformers.util.exceptions import NotFoundError


def test_api_detects_if_cached():
    api = StartupRadarAPI("demo", session_factory=requests.Session)
    assert api.is_cached is False


def test_api_404_raises():
    # I mock it all
    # I mock it all
    # I mock it all
    # and I mock it now...
    # https://www.youtube.com/watch?v=hFDcoX7s6rE

    mock_response = Mock()
    mock_response.status_code = 404

    mock_session = Mock()
    mock_session.get = Mock(return_value=mock_response)

    mock_session_factory = lambda: mock_session
    api = StartupRadarAPI("wront", session_factory=mock_session_factory)
    with pytest.raises(NotFoundError):
        api.get_whois("willnotexist.com")


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
