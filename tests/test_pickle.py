import joblib
import pytest

from startupradar.transformers import core, basic, pandas

TRANSFORMERS = [
    # basic
    basic.ColumnPrefixTransformer("_"),
    basic.CommonStringTransformer(),
    basic.CounterTransformer(),
    basic.DomainNameTransformer(),
    # core
    core.LinkTransformer(None),
    core.BacklinkTransformer(None),
    core.BacklinkTypeCounter(None),
    core.DomainTextTransformer(None),
    core.WhoisTransformer(None),
    # pandas
    pandas.ColumnTransformerDF([]),
    pandas.CountVectorizerDF(),
    pandas.FeatureUnionDF([]),
    pandas.OneHotEncoderDF(),
    pandas.PipelineDF([]),
    pandas.TfidfVectorizerDF(),
    pandas.Vectorizer(),
]


@pytest.mark.parametrize("transformer", TRANSFORMERS)
def test_pickle(transformer, tmp_path):
    joblib.dump(transformer, tmp_path.joinpath("transformer.pkl"))
    transformer = joblib.load(tmp_path.joinpath("transformer.pkl"))
    assert transformer is not None
