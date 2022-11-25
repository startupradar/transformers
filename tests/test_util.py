import pytest

from startupradar.transformers.api import InvalidDomainError
from startupradar.transformers.util import DomainNameTransformer

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
