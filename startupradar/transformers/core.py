"""
Core transformers mapping directly to API functionality.
"""
import logging
from abc import ABC
from collections import Counter
from datetime import datetime
from functools import cached_property, lru_cache
from random import shuffle

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from startupradar.transformers.basic import CounterTransformer
from startupradar.transformers.config import TransformerConfig
from startupradar.transformers.util.api import StartupRadarAPI, get_text_or_empty_dict
from startupradar.transformers.util.exceptions import NotFoundError

N_DEFAULT = 10


class ApiTransformer(TransformerMixin):
    def __init__(self, api: StartupRadarAPI):
        self.api = api

    def get_params(self, deep=False):
        # this returns the api used when initializing
        # necessary to make GridSearchVC work without setting TransformerConfig.api
        return {"api": self.api}

    def set_params(self, params):
        self.api = params["api"]

    def __getstate__(self):
        # this data is used for pickling
        # returns everything except the api
        state = {k: v for k, v in self.__dict__.items() if k != "api"}
        logging.debug(f"returning state ({self=}, {state=})")
        return state

    def __setstate__(self, state):
        # this gets used to restore the transformer when pickling
        # sets all the attributes and switches out the API

        logging.debug(f"unpickling ({state=})")
        for attr, value in state.items():
            setattr(self, attr, value)

        logging.debug(f"switching out api during unpickling ({TransformerConfig.api=})")
        if not TransformerConfig.api:
            raise RuntimeError(
                "You're unpickling an API transformer without setting TransformerConfig.api, "
                "please assign the desired API wrapper to TransformerConfig.api upfront"
            )

        self.api = TransformerConfig.api


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

    def __init__(self, api, n: int = N_DEFAULT):
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
        # shuffle to leverage multi-core pipeline with caching
        domains = list(domains)
        shuffle(domains)

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

        if self.api.is_cached:
            # shuffle to fetch parallelized
            domains = list(X)
            shuffle(domains)
            map(get_text_or_empty_dict, domains)

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


class BacklinkTypeCounter(CounterTransformer):
    def __init__(self, api: StartupRadarAPI):
        super().__init__()
        self.api = api

    def create_counter(self, domain: str) -> Counter:
        links = self.api.get_backlinks(domain)
        return Counter([self.get_type(link["domain"]) for link in links])

    @lru_cache
    def get_type(self, domain):
        return self.type_per_domain.get(domain, "unknown")

    @cached_property
    def type_per_domain(self) -> dict:
        return {s["domain"]: s["category"] for s in self.api.get_sources()}


class WhoisTransformer(SeriesTransformer):
    """
    Transformer to add whois-based information.
    """

    COLUMNS_DEFAULT = ("created", "changed", "expires")
    COLUMNS_AGE = ("days_since_created", "days_since_changed")

    def __init__(
        self, api, add_timestamps=True, add_ages=True, today: datetime.date = None
    ):
        super().__init__(api)
        self.add_timestamps = add_timestamps
        self.add_ages = add_ages

        self.today = datetime.today()
        if today:
            self.today = today

    def transform(self, X, y=None):
        def get_age_or_none(d: datetime.date):
            if not d:
                return None

            return (datetime.today() - d).total_seconds() / 60 / 60 / 24

        def get_whois_or_empty(domain: str):
            try:
                return self.api.get_whois(domain)
            except NotFoundError:
                # return columns to ensure df has all columns, even if no whois exists
                return {"created": None, "changed": None, "expires": None}

        whoises = [get_whois_or_empty(d) for d in X.tolist()]
        df_whoises = pd.DataFrame(whoises, index=X)

        if self.add_ages:
            df_whoises["days_since_created"] = df_whoises["created"].apply(
                get_age_or_none
            )
            df_whoises["days_since_changed"] = df_whoises["changed"].apply(
                get_age_or_none
            )

        # ensure ordered columns, since dict keys could be scrambled
        return df_whoises[self.get_feature_names_out()]

    def get_params(self, deep=False):
        return {"api": self.api, "add_ages": self.add_ages, "today": self.today}

    def get_feature_names_out(self, feature_names_in=None):
        # output must be list in order to be used as columns index
        columns_out = []

        if self.add_timestamps:
            columns_out.extend(self.COLUMNS_DEFAULT)

        if self.add_ages:
            columns_out.extend(self.COLUMNS_AGE)

        return columns_out


class SocialsTransformer(SeriesTransformer):
    """
    Uses the socials endpoint to generate a dataframe.
    """

    def transform(self, X, y=None):
        def get_socials_or_none(domain):
            # return empty dict to avoid indexing errors
            try:
                socials = self.api.get_socials(domain)
                return socials if socials else {}
            except NotFoundError:
                return {}

        # make socials dataframe
        socials = X.apply(get_socials_or_none)
        df_socials = pd.DataFrame.from_records(socials, index=X.to_list())

        # check that all columns exist
        df = pd.DataFrame(index=X)
        for col in df_socials.columns:
            df[f"has_{col}"] = df_socials[col].notna()
        return df
