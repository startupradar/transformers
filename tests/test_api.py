from unittest.mock import Mock

import pytest

from startupradar.transformers.util.api import StartupRadarAPI
from startupradar.transformers.util.exceptions import NotFoundError


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


def test_api_filter_without_homepage_does_not_raise(monkeypatch):
    api = StartupRadarAPI(None)

    def mock_request(endpoint, params=None):
        if endpoint == "/web/domains/test.de":
            return {"domain": "test.de"}
        elif endpoint == "/web/domains/test.de/homepage":
            raise NotFoundError()
        else:
            raise RuntimeError(f"unmocked request: ({endpoint=})")

    monkeypatch.setattr(api, "_request", mock_request)

    filtered = list(api.filter_without_homepage(["test.de"]))
    assert filtered == []
