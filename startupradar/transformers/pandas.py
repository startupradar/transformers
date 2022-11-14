"""
sklearn transformers adapted to work on pandas DataFrames or Series'.
"""
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline


class FeatureUnionDF(TransformerMixin):
    """
    Feature Union that leverages pandas dataframes.
    Requires transformers to implement get_feature_names.
    """

    def __init__(self, *args, **kwargs):
        self.fu = FeatureUnion(*args, **kwargs)

    def fit(self, X, y=None, **kwargs):
        self.fu.fit(X, y, **kwargs)
        return self

    def transform(self, X, y=None, **fit_params):
        return pd.DataFrame(
            self.fu.transform(X), columns=self.fu.get_feature_names_out()
        )

    def get_feature_names_out(self, feature_names_in=None):
        return self.fu.get_feature_names_out(feature_names_in)

    def set_params(self, **kwargs):
        self.fu.set_params(**kwargs)

    def get_params(self, deep=False):
        return self.fu.get_params(deep)


class PipelineDF(Pipeline):
    def get_feature_names_out(self, feature_names_in=None):
        return self.steps[-1][1].get_feature_names_out(feature_names_in)
