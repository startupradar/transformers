import logging

import joblib
import pandas as pd
import pytest
from minimalkv.memory import DictStore
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

from startupradar.transformers import core, basic, pandas
from startupradar.transformers.config import TransformerConfig
from startupradar.transformers.helpers import TurnToIter
from startupradar.transformers.util.api import StartupRadarAPI, MinimalKeyValueCache

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
    TransformerConfig.api = StartupRadarAPI(None)
    joblib.dump(transformer, tmp_path.joinpath("transformer.pkl"))
    transformer = joblib.load(tmp_path.joinpath("transformer.pkl"))
    assert transformer is not None


@pytest.mark.vcr
def test_pickle_pipeline(api, tmp_path):
    TransformerConfig.api = None

    pipeline = pandas.PipelineDF(
        [
            ("get", core.DomainTextTransformer(api=api)),
            ("iter", TurnToIter()),
            ("tfidf", pandas.TfidfVectorizerDF(max_features=10)),
            (
                "union",
                pandas.FeatureUnionDF(
                    [
                        ("svd", TruncatedSVD(n_components=3)),
                        ("kbest", SelectKBest(k=1)),
                    ]
                ),
            ),
            ("estimator", DecisionTreeClassifier(random_state=123)),
        ]
    )

    # make sure that pipeline is unfitted if new
    with pytest.raises(NotFittedError):
        pipeline.predict(pd.Series(["startupradar.co"]))

    # set up df with startupradar as True and karllorey as False
    domains = ["karllorey.com"] * 20 + ["startupradar.co"] * 20
    df = pd.DataFrame(
        [(d, d == "startupradar.co") for d in domains], columns=["domain", "target"]
    )

    # wrap pipeline in gridsearch and fit
    params = {"tfidf__max_features": [10, 11, 12]}
    pipeline = RandomizedSearchCV(
        pipeline, params, cv=2, n_jobs=1, n_iter=1, random_state=123
    )
    pipeline.fit(df["domain"], df["target"])

    # pickle fitted pipeline
    tmp_file = tmp_path.joinpath("dump.pkl")
    joblib.dump(pipeline, tmp_file)

    # unpickle fails without TransformerConfig
    with pytest.raises(RuntimeError):
        logging.info(f"{TransformerConfig.api=}")
        gs = joblib.load(tmp_file)
        gs.predict(df["domain"])

    # but works after being set
    TransformerConfig.api = api
    gs = joblib.load(tmp_file)
    gs.predict(df["domain"])
