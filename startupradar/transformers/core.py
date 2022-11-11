"""
Core transformers mapping directly to API functionality.
"""
from abc import ABC

import pandas as pd
from sklearn.base import TransformerMixin

from startupradar.transformers.api import StartupRadarAPI


class ApiTransformer(TransformerMixin):
    def __init__(self, api: StartupRadarAPI):
        self.api = api


class SeriesTransformer(ApiTransformer, ABC):
    """
    Transforms a series.
    """

    def fit(self, X, y=None):
        assert isinstance(
            X, pd.Series
        ), "Transformer works on pandas.Series of domains(str)"
        return self

    def transform(self, X, y=None):
        assert isinstance(
            X, pd.Series
        ), "Transformer works on pandas.Series of domains(str)"

    def get_params(self, deep):
        return None


class LinkTransformer(SeriesTransformer):
    """
    Creates columns for all domains the given domain links to.
    """

    def __init__(self, api):
        super().__init__(api)
        self.columns = None
        self._domains = None

    def fit(self, X, y=None):
        super().fit(X, y)
        self._domains = list({b for a, b in self._fetch_tuples(X)})
        return self

    def transform(self, X, y=None):
        super().transform(X, y)
        assert self._domains is not None, 'not fitted'

        backlinks = self._fetch_tuples(X)

        rows = []
        for domain_given in X:
            for domain_link in self._domains:
                rows.append((domain_given, domain_link, int((domain_given, domain_link) in backlinks)))

        df = pd.DataFrame(rows, columns=["domain", "link", 'v'])

        df_links = df.pivot(index="domain", columns="link", values="v")

        # ensure proper indexing
        df_links = pd.DataFrame(index=X).join(df_links).fillna(0).astype(int)[self._domains]

        return df_links

    def _fetch_tuples(self, domains):
        tuples = []
        for domain in domains:
            for link in self._fetch_links(domain):
                tuples.append((domain, link["domain"]))
        return tuples

    def _fetch_links(self, domain: str):
        return self.api.get_links(domain)

    def get_feature_names_out(self, feature_names_in=None):
        assert self._domains is not None, 'not fitted'
        return self._domains


class BacklinkTransformer(LinkTransformer):
    """
    Creates columns for all domains that link to the given domain.
    """

    def _fetch_links(self, domain: str):
        # overwrite method to fetch backlinks instead of links
        return self.api.get_backlinks(domain)


class DomainTextTransformer(SeriesTransformer):
    """
    Retrieve the text for the given domains.
    """

    def transform(self, X, y=None):
        super().transform(X, y)
        return X.apply(self._fetch_text)

    def _fetch_text(self, domain: str):
        return self.api.get_text(domain)["html_body_text"]
