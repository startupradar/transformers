"""
Core transformers mapping directly to API functionality.
"""
import logging
from abc import ABC
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from startupradar.transformers.util.api import StartupRadarAPI
from startupradar.transformers.util.exceptions import NotFoundError


class ApiTransformer(TransformerMixin):
    def __init__(self, api: StartupRadarAPI):
        self.api = api

    def get_params(self, deep):
        return {"api": self.api}


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


class LinkTransformer(SeriesTransformer):
    """
    Creates columns for all domains the given domain links to.
    """

    def __init__(self, api, n: int = 10):
        super().__init__(api)
        self._domains = None
        self.n = n

    def fit(self, X, y=None):
        assert isinstance(X, pd.Series)

        domains = X.values.tolist()

        # count occurrences
        counter = Counter((b for a, b in self._gen_tuples(domains)))

        # add top n
        self._domains = [domain for domain, count in counter.most_common(self.n)]

        if not self._domains:
            logging.warning("no common links found")

        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.Series)
        assert self._domains is not None, "not fitted"

        domains = X.values.tolist()

        # todo make this work with sparse matrices and columns
        df = pd.DataFrame(
            np.zeros([len(domains), len(self._domains)]),
            index=domains,
            columns=self._domains,
            dtype=bool,
        )

        for d_from, d_to in self._gen_tuples(set(domains)):
            if d_to in self._domains:
                df.at[d_from, d_to] = True

        return df

    def _gen_tuples(self, domains):
        for domain in domains:
            for link in self._fetch_links(domain):
                yield domain, link["domain"]

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
