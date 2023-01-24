from unittest.mock import Mock

import pytest
import requests

from startupradar.transformers.util.api import StartupRadarAPI


@pytest.fixture
def api():
    # set wrong api key to raise errors
    return StartupRadarAPI(api_key="wrong", session_factory=requests.Session)


def domain_to_dict(domain: str):
    return {"domain": domain}


DOMAIN_HYPERLINKS = [
    ("karllorey.com", "google.com"),
    ("karllorey.com", "crunchbase.com"),
    ("startupradar.co", "google.com"),
    ("startupradar.co", "crunchbase.com"),
    ("startupradar.co", "pitchbook.com"),
    ("crunchbase.com", "google.com"),
    ("irrelevant.com", "google.com"),
]


@pytest.fixture
def mock_api():
    mock_api = Mock()

    mock_domains = ["karllorey.com", "startupradar.co"]
    mock_api.get_domains = [domain_to_dict(d) for d in mock_domains]

    mock_api.get_links = lambda d: [
        domain_to_dict(d_to) for d_from, d_to in DOMAIN_HYPERLINKS if d_from == d
    ]

    mock_api.get_backlinks = lambda d: [
        domain_to_dict(d_from) for d_from, d_to in DOMAIN_HYPERLINKS if d_to == d
    ]

    mock_api.get_sources = lambda: [
        {"domain": "crunchbase.com", "category": "platform"},
        {"domain": "karllorey.com", "category": "person"},
    ]

    mock_api.get_socials = lambda domain: {
        "twitter_url": f"https://twitter.com/{domain}",
        "facebook_url": f"https://facebook.com/{domain}",
        "linkedin_url": f"https://linkedin.com/company/{domain}",
        "crunchbase_url": f"https://crunchbase.com/organization/{domain}",
        "instagram_url": f"https://instagram.com/{domain}",
        "email": f"mail@{domain}",
    }
    mock_api.get_text = lambda d: {"html_meta_description": "Bla bla bla"}

    return mock_api
