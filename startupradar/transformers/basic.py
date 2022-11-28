"""
Utility transformers without API-specific functionality.
"""
import pandas as pd
import tldextract
from sklearn.base import TransformerMixin

from startupradar.transformers.api import InvalidDomainError
from startupradar.transformers.pandas import CountVectorizerDF


class DomainNameTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        tlds = [tldextract.extract("http://" + domain) for domain in X]
        df = pd.DataFrame(tlds, index=X)

        # check inputs
        for row, domain_input in zip(df.to_dict(orient="records"), X):
            domain_parsed = ".".join([row["domain"], row["suffix"]])
            if domain_parsed != domain_input:
                raise InvalidDomainError(
                    f"parsed domain looks off: {domain_parsed=}, {domain_input=}"
                )

        # create lengths
        df["length"] = df["domain"].apply(len)
        return df[["suffix", "length"]]

    def get_feature_names_out(self, feature_names_in=None):
        return ["suffix", "length"]


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


class CommonStringTransformer(TransformerMixin):
    """
    Computes common strings from passed data.
    Internally, this uses the CountVectorizer from scikit-learn.
    """

    def __init__(self, **kwargs):
        self.vec = CountVectorizerDF(
            analyzer="char",
            **kwargs,
        )

    def fit(self, X, y=None):
        self.vec.fit(X, y)
        return self

    def transform(self, X):
        return self.vec.transform(X)

    def get_feature_names_out(self, feature_names_in=None):
        return self.vec.get_feature_names_out(feature_names_in)
