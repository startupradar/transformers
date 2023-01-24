from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from startupradar.transformers.core import (
    LinkTransformer,
    BacklinkTransformer,
    BacklinkTypeCounter,
    WhoisTransformer,
)


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


def test_link_type_counter(mock_api):
    t = BacklinkTypeCounter(mock_api)
    t.fit(["google.com"])
    out = t.transform(["google.com"])
    assert out["person"].tolist() == [1]
    assert out["platform"].tolist() == [1]
    assert out["unknown"].tolist() == [2]


@pytest.mark.vcr
def test_whois_transformer(api):
    t = WhoisTransformer(api, today=datetime(2023, 1, 1))
    domain_series = pd.Series(["startupradar.co", "karllorey.com", "karllorey.de"])
    df_out = t.transform(domain_series)
    print(df_out)

    # assert index matches
    assert df_out.index.to_list() == domain_series.tolist()

    # assert has all columns
    assert df_out.columns.tolist() == [
        "created",
        "changed",
        "expires",
        "days_since_created",
        "days_since_changed",
    ]

    # check datatype
    for col in ["created", "changed", "expires"]:
        assert df_out[col].dtype == np.dtype("datetime64[ns]")
