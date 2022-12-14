import pytest

from startupradar.transformers.export import DomainExport


@pytest.mark.vcr
def test_export(mock_api):
    exporter = DomainExport(mock_api)

    input_domains = ["startupradar.co", "crunchbase.com"]
    df = exporter.create_for_domains(input_domains)
    print(df.to_string())

    assert len(df) == len(input_domains)

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
        "backlinks_person",
        "backlinks_unknown",
    ]

    # check columns
    row = df.loc["startupradar.co"]
    assert row["url"] == "https://startupradar.co"
    assert row["backlinks_unknown"] == 0
