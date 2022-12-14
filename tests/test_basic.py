from collections import Counter

import pandas as pd
import pytest

from startupradar.transformers.util.exceptions import InvalidDomainError
from startupradar.transformers.basic import (
    DomainNameTransformer,
    CommonStringTransformer,
    CounterTransformer,
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


def test_counter_transformer():
    class CharacterCount(CounterTransformer):
        def create_counter(self, x):
            # count characters
            return Counter(list(x))

    out = CharacterCount().fit_transform(["aaa", "bbb", "abc"])

    assert set(out.columns) == set("abc")
    assert out["a"].values.tolist() == [3, 0, 1]
    assert out["b"].values.tolist() == [0, 3, 1]
    assert out["c"].values.tolist() == [0, 0, 1]
