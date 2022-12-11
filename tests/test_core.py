from unittest.mock import Mock

import pandas as pd
import pytest

from startupradar.transformers.core import LinkTransformer


def domain_to_dict(domain: str):
    return {"domain": domain}


@pytest.fixture
def mock_api():
    mock_api = Mock()

    mock_domains = ["karllorey.com", "startupradar.co"]
    mock_api.get_domains = [domain_to_dict(d) for d in mock_domains]

    mock_links = {
        "karllorey.com": [
            "google.com",
            "crunchbase.com",
        ],
        "startupradar.co": [
            "google.com",
            "crunchbase.com",
            "pitchbook.com",
        ],
        "crunchbase.com": ["google.com"],
    }
    mock_api.get_links = lambda d: [
        domain_to_dict(linked_domain) for linked_domain in mock_links[d]
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
