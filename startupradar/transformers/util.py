"""
Utility transformers without API-specific functionality.
"""
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
import tldextract


class DomainNameTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        tlds = [tldextract.extract("http://" + domain).suffix for domain in X]
        return pd.DataFrame(tlds, index=X, columns=["tld"])

    def get_feature_names_out(self, feature_names_in=None):
        return ["tld"]


class ColumnPrefixTransformer(TransformerMixin):
    def __init__(self, prefix: str):
        self.prefix = prefix
        self.columns = None

    def fit(self, X, y=None):
        self.columns = list(X.columns.values)
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        assert list(X.columns.values) == self.columns
        X.columns = self.get_feature_names_out()
        return X

    def get_feature_names_out(self, feature_names_in=None):
        return [f"{self.prefix}_{c}" for c in self.columns]


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
