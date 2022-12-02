"""
Classes used to export human-readable data.
"""
import logging
import typing
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import pandas as pd

from startupradar.transformers.util.api import StartupRadarAPI
from startupradar.transformers.util.exceptions import NotFoundError


@lru_cache
def get_text_or_none(api, domain) -> dict:
    # needs caching as it gets called from different places
    # which would result in several /text calls
    try:
        return api.get_text(domain)
    except NotFoundError:
        return {}


def tqdm_if_installed(iterable, total=None):
    """
    Passes to tqdm if installed.
    """

    try:
        import tqdm

        yield from tqdm(iterable, total=total)
    except ModuleNotFoundError:
        yield from iterable


class DomainExport:
    THREAD_COUNT = 8
    META_DESCRIPTION_CUTOFF = 100

    def __init__(self, api: StartupRadarAPI, scorers: typing.Optional[dict] = None):
        self.api = api
        self.scorers = scorers if scorers else {}
        self._use_cache_warming = self.api.is_cached

    def _warm_cache(self, method, domains):
        if self._use_cache_warming:
            logging.info(f"warming cache {method=}")
            with ThreadPoolExecutor(self.THREAD_COUNT) as e:
                results = e.map(method, domains)
                for _ in tqdm_if_installed(results):
                    pass
            logging.info("cache on fire")

    def create_for_domains(self, domains: list):
        df_domains = pd.DataFrame(domains, columns=["domain"])
        df_domains["url"] = df_domains["domain"].apply(
            lambda domain: "https://" + domain
        )
        df_domains["description"] = df_domains["domain"].apply(
            self._get_meta_description_capped
        )
        df_domains = df_domains.set_index("domain")

        #
        # socials
        #
        self._warm_cache(self.api.get_socials, domains)
        socials = [self.api.get_socials(domain) for domain in domains]
        df_socials = pd.DataFrame(socials)
        df_socials.index = domains

        #
        # scorers
        #
        for scorer_name, scorer_callable in self.scorers.items():
            df_scorer = pd.DataFrame(scorer_callable(domains))
            if len(df_scorer) != len(domains):
                raise RuntimeError("scorer returned less rows")

            df_scorer.columns = [f"scorer_{scorer_name}"]
            df_scorer.index = domains
            df_domains = pd.concat([df_domains, df_scorer], axis=1)

        #
        # backlinks
        #
        df_backlinks = self.make_df_backlinks(domains)

        df_full = pd.concat([df_domains, df_socials, df_backlinks], axis=1)

        return df_full

    def make_df_backlinks(self, domains):
        self._warm_cache(self.api.get_backlinks, domains)

        domain_to_sourcetype = {
            s["domain"]: s["category"] for s in self.api.get_sources()
        }
        df_backlinks = pd.DataFrame(index=domains)
        backlinks_per_domain = [
            set(map(lambda b: b["domain"], self.api.get_backlinks(d))) for d in domains
        ]
        backlinks_counter_per_domains = [
            Counter(map(lambda d: domain_to_sourcetype.get(d), backlinks))
            for backlinks in backlinks_per_domain
        ]
        df_backlinks["backlinks_all"] = pd.Series(
            [len(b) for b in backlinks_per_domain], index=domains
        )
        for col, category in {
            "investors": "Investors",
            "academia": "Academia",
            "accelerators": "Accelerators",
            "news": "News",
        }.items():
            df_backlinks[f"backlinks_{col}"] = pd.Series(
                [c.get(category) or 0 for c in backlinks_counter_per_domains],
                index=domains,
            )
        return df_backlinks

    def _get_meta_description_capped(self, domain: str):
        result = get_text_or_none(self.api, domain)

        meta_desc = result.get("html_meta_description", None)
        if meta_desc and len(meta_desc) > self.META_DESCRIPTION_CUTOFF:
            meta_desc = meta_desc[: self.META_DESCRIPTION_CUTOFF] + "..."

        return meta_desc
