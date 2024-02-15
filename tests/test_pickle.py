import logging

import joblib
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from startupradar.transformers import core, basic, pandas
from startupradar.transformers.config import TransformerConfig
from startupradar.transformers.helpers import TurnToIter
from startupradar.transformers.util.api import StartupRadarAPI


class NotPickleableAPI(StartupRadarAPI):
    """
    Manually make this API not pickleable to ensure it's never tried to be pickled.
    """

    def __init__(self):
        super().__init__("")

    def __getstate__(self):
        raise RuntimeError("this object got pickled")

    def __setstate__(self, state):
        raise RuntimeError("this object got un-pickled")


TRANSFORMERS = [
    # basic
    basic.ColumnPrefixTransformer("_"),
    basic.CommonStringTransformer(),
    basic.CounterTransformer(),
    basic.DomainNameTransformer(),
    # core
    core.LinkTransformer(NotPickleableAPI()),
    core.BacklinkTransformer(NotPickleableAPI()),
    core.BacklinkTypeCounter(NotPickleableAPI()),
    core.DomainTextTransformer(NotPickleableAPI()),
    core.WhoisTransformer(NotPickleableAPI()),
    # pandas
    pandas.ColumnTransformerDF([]),
    pandas.CountVectorizerDF(),
    pandas.FeatureUnionDF([]),
    pandas.OneHotEncoderDF(),
    pandas.PipelineDF([("test", ColumnTransformer([], remainder="passthrough"))]),
    pandas.TfidfVectorizerDF(),
    # pandas.Vectorizer(),
]


@pytest.mark.parametrize("transformer", TRANSFORMERS)
def test_pickle(transformer, tmp_path):
    pkl_file_path = tmp_path.joinpath("transformer.pkl")

    # dump
    joblib.dump(transformer, pkl_file_path)

    # load again
    TransformerConfig.api = NotPickleableAPI()
    transformer = joblib.load(pkl_file_path)

    assert transformer is not None


@pytest.mark.vcr
def test_pickle_pipeline(api, tmp_path):
    TransformerConfig.api = api
    pipeline = pandas.PipelineDF(
        [
            ("get", core.DomainTextTransformer()),
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

    # but works after being set
    TransformerConfig.api = api
    gs = joblib.load(tmp_file)
    gs.predict(df["domain"])


class FakePipeline(Pipeline):
    def score(self, **kwargs):
        return 0


@pytest.mark.parametrize(
    "transformer", TRANSFORMERS, ids=map(lambda t: type(t), TRANSFORMERS)
)
def test_gridsearch(transformer, tmp_path):
    tmp_file = tmp_path.joinpath("pipeline.pkl")
    assert not tmp_file.is_file()
    pipeline = FakePipeline(
        [
            ("transformer", transformer),
        ]
    )

    TransformerConfig.api = NotPickleableAPI()
    gs = GridSearchCV(pipeline, {}, n_jobs=2, cv=2)
    try:
        gs.fit(pd.DataFrame(range(10)))
    except ValueError:
        pass
    # joblib.dump(pipeline, tmp_file)
