"""
sklearn transformers adapted to work on pandas DataFrames or Series'.
"""
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder


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
        out = self.fu.transform(X)
        return pd.DataFrame(out, columns=self.fu.get_feature_names_out())

    def get_feature_names_out(self, feature_names_in=None):
        return self.fu.get_feature_names_out(feature_names_in)

    def set_params(self, **kwargs):
        self.fu.set_params(**kwargs)

    def get_params(self, deep=False):
        return self.fu.get_params(deep)


class PipelineDF(Pipeline):
    def get_feature_names_out(self, feature_names_in=None):
        return self.steps[-1][1].get_feature_names_out(feature_names_in)


class Vectorizer(TransformerMixin):
    """
    Vectorizer leveraging composition of an sklearn vectorizer to produce dataframe outputs.
    """

    vec = None

    def fit(self, docs, y=None):
        self.vec.fit(docs, y)
        return self

    def transform(self, docs):
        out_raw = self.vec.transform(docs)
        df = pd.DataFrame(out_raw.todense(), columns=self.vec.get_feature_names_out())
        return df

    def get_feature_names_out(self, feature_names_in=None):
        return self.vec.get_feature_names_out(input_features=feature_names_in)

    def set_params(self, **params: dict):
        self.vec.set_params(**params)

    def get_params(self, deep=False):
        return self.vec.get_params(deep=deep)


class TfidfVectorizerDF(Vectorizer):
    def __init__(self, **kwargs):
        self.vec = TfidfVectorizer(**kwargs)


class CountVectorizerDF(Vectorizer):
    def __init__(self, **kwargs):
        self.vec = CountVectorizer(**kwargs)


class OneHotEncoderDF(TransformerMixin):
    def __init__(self, **kwargs):
        kwargs["sparse"] = False
        self.enc = OneHotEncoder(**kwargs)

    def fit(self, X: pd.DataFrame, y=None):
        self.enc.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame):
        transformed = self.enc.transform(X)
        columns = self.enc.get_feature_names_out(X.columns)
        return pd.DataFrame(transformed, columns=columns)

    def get_params(self, deep=False):
        return self.enc.get_params(deep)

    def set_params(self, **params: dict):
        return self.enc.set_params(**params)

    def get_feature_names_out(self, features_input=None):
        return self.enc.get_feature_names_out(features_input)


class ColumnTransformerDF(TransformerMixin):
    def __init__(self, transformers: list, **kwargs):
        # disable sparse
        kwargs["sparse_threshold"] = 0

        self.trans = ColumnTransformer(transformers, **kwargs)

    def fit(self, X, y=None):
        self.trans.fit(X, y)
        return self

    def transform(self, X):
        trans = self.trans.transform(X)
        columns = self.trans.get_feature_names_out(X.columns.values)
        return pd.DataFrame(trans, columns=columns)

    def get_params(self, deep=False):
        return self.trans.get_params(deep)

    def set_params(self, **kwargs):
        return self.trans.set_params(**kwargs)

    def get_feature_names_out(self, input_features=None):
        return self.trans.get_feature_names_out(input_features)
