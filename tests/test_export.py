import pytest

from startupradar.transformers.export import DomainExport


@pytest.mark.vcr
def test_export(api):
    exporter = DomainExport(api)

    input_domains = ["startupradar.co"]
    df = exporter.create_for_domains(input_domains)
    print(df.to_string())

    assert len(df) == len(input_domains)

    row = df.loc["startupradar.co"]
    print(row)

    # check layout
    assert df.columns.values.tolist() == [
        "url",
        "description",
        "twitter_url",
        "facebook_url",
        "linkedin_url",
        "crunchbase_url",
        "instagram_url",
        "email",
        "backlinks_all",
        "backlinks_investors",
        "backlinks_academia",
        "backlinks_accelerators",
        "backlinks_news",
    ]

    # check columns
    assert row["url"] == "https://startupradar.co"

    assert row["backlinks_all"] == 2
    assert row["backlinks_investors"] == 0
    assert row["backlinks_academia"] == 0
    assert row["backlinks_accelerators"] == 0
    assert row["backlinks_news"] == 0
