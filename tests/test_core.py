from unittest.mock import Mock

import pandas as pd
import pytest

from startupradar.transformers.core import LinkTransformer, BacklinkTransformer


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

    return mock_api


def test_link_transformer(mock_api):
    t = LinkTransformer(mock_api, n=2)

    domains = ["karllorey.com", "startupradar.co", "crunchbase.com"]
    series_domains = pd.Series(domains)
    t.fit(series_domains)
    print(t._domains)

    df = t.transform(series_domains)
    assert df.columns.tolist() == ["google.com", "crunchbase.com"]
    assert df["google.com"].tolist() == [True, True, True]
    assert df["crunchbase.com"].tolist() == [True, True, False]


def test_backlink_transformer(mock_api):
    t = BacklinkTransformer(mock_api, n=5)

    domains = ["crunchbase.com", "pitchbook.com"]
    series_domains = pd.Series(domains)
    t.fit(series_domains)
    print(t._domains)

    df = t.transform(series_domains)
    assert df.columns.tolist() == ["startupradar.co", "karllorey.com"]
    assert df["startupradar.co"].tolist() == [True, True]
    assert df["karllorey.com"].tolist() == [True, False]
