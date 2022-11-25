import pytest

from startupradar.transformers.api import InvalidDomainError
from startupradar.transformers.util import (
    DomainNameTransformer,
    CommonStringTransformer,
)

DOMAINS_INVALID = [
    "127.0.0.1",
    "localhost",
    "api.startupradar.co",
    "www.startupradar.co",
]


def test_domain_name_transformer_lengths():
    t = DomainNameTransformer()
    df = t.transform(["karllorey.com", "startupradar.co"])
    assert df["length"].values.tolist() == [9, 12]


@pytest.mark.parametrize("invalid_domain", DOMAINS_INVALID)
def test_domain_name_transformer_raises(invalid_domain):
    t = DomainNameTransformer()
    with pytest.raises(InvalidDomainError):
        t.transform([invalid_domain])


def test_domain_name_transformer_feature_names_out():
    t = DomainNameTransformer()
    df = t.transform(["karllorey.com", "startupradar.co"])
    assert df.columns.tolist() == t.get_feature_names_out()


def test_common_string_transformer():
    t = CommonStringTransformer(max_df=1.0, min_df=0.5, ngram_range=(3, 5))
    domains = ["kit.edu", "hs-ka.de", "rwth-aachen.de", "rwth-campus.com"]
    df_out = t.fit_transform(domains)
    common_substrings = [
        ".de",
        "rwt",
        "rwth",
        "rwth-",
        "th-",
        "wth",
        "wth-",
    ]
    assert df_out.columns.values.tolist() == common_substrings
