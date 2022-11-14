"""
Core transformers mapping directly to API functionality.
"""
import logging
from abc import ABC
from collections import Counter

import pandas as pd
from sklearn.base import TransformerMixin

from startupradar.transformers.api import StartupRadarAPI, NotFoundError


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

    CUTOFF_BELOW = 2

    def __init__(self, api):
        super().__init__(api)
        self.columns = None
        self._domains = None

    def fit(self, X, y=None):
        assert isinstance(X, pd.Series)

        domains = X.values.tolist()

        # count occurrences
        counter = Counter((b for a, b in self._fetch_tuples(domains)))

        # add above threshold
        self._domains = list(
            {domain for domain, count in counter.items() if count >= self.CUTOFF_BELOW}
        )
        if not self._domains:
            logging.warning("no common links found")

        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.Series)
        assert self._domains is not None, "not fitted"

        domains = X.values.tolist()
        backlinks = self._fetch_tuples(set(domains))

        rows = []
        for domain_given in domains:
            for domain_link in self._domains:
                triple = (
                    domain_given,
                    domain_link,
                    # is there a link?
                    int((domain_given, domain_link) in backlinks),
                )
                rows.append(triple)

        df = pd.DataFrame(rows, columns=["domain", "link", "v"])

        df_links = df.pivot(index="domain", columns="link", values="v")

        # ensure proper indexing
        df_out = pd.DataFrame(index=X).join(df_links)[self._domains]

        return df_out

    def _fetch_tuples(self, domains):
        tuples = []
        for domain in domains:
            for link in self._fetch_links(domain):
                tuples.append((domain, link["domain"]))
        return tuples

    def _fetch_links(self, domain: str):
        return self.api.get_links(domain)

    def get_feature_names_out(self, feature_names_in=None):
        assert self._domains is not None, "not fitted"
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
        assert isinstance(X, pd.Series)
        series = X.apply(self._fetch_text)
        df = pd.DataFrame(series)
        df.columns = ["text"]
        return df

    def _fetch_text(self, domain: str):
        assert isinstance(domain, str), f"domain is not a string, {type(domain)=} given"
        try:
            return self.api.get_text(domain)["html_body_text"]
        except NotFoundError:
            return ""

    def get_feature_names_out(self, feature_names_in=None):
        return ["text"]
