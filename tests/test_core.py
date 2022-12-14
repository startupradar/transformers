import pandas as pd

from startupradar.transformers.core import (
    LinkTransformer,
    BacklinkTransformer,
    BacklinkTypeCounter,
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
