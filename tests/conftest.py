import pytest
import requests

from startupradar.transformers.util.api import StartupRadarAPI


@pytest.fixture
def api():
    return StartupRadarAPI(api_key="demo", session_factory=requests.Session)
