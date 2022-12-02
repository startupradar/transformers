import requests

from startupradar.transformers.util.api import StartupRadarAPI


def test_api_detects_if_cached():
    api = StartupRadarAPI("demo", session_factory=requests.Session)
    assert api.is_cached is False
