"""
Classes to access the StartupRadar API.
"""
import json
import logging
from abc import ABC
from itertools import chain
from urllib.parse import urljoin, quote

import requests
import tldextract
from minimalkv import KeyValueStore

from startupradar.transformers.util.exceptions import (
    StartupRadarAPIError,
    NotFoundError,
    ForbiddenError,
    InvalidDomainError,
    NotInCacheException,
)

DOMAINS_IGNORED_BACKLINKS = (
    "google.com",
    "facebook.com",
    "twitter.com",
    "instagram.com",
    "linkedin.com",
    "youtube.com",
    "apple.com",
)


class CacheI(ABC):
    def get(self, endpoint: str):
        raise NotImplementedError()

    def put(self, endpoint: str, result):
        raise NotImplementedError()


class PassThroughCache(CacheI):
    def get(self, endpoint: str):
        raise NotInCacheException()

    def put(self, endpoint: str, result):
        return


class MinimalKeyValueCache(CacheI):
    def __init__(self, store: KeyValueStore):
        self.store = store

    def endpoint_to_key(self, endpoint: str):
        return quote(endpoint, safe="")

    def get(self, endpoint: str):
        try:
            key = self.endpoint_to_key(endpoint)
            cache_bytes = self.store.get(key)
            return json.loads(cache_bytes.decode())
        except KeyError:
            raise NotInCacheException from KeyError

    def put(self, endpoint: str, result):
        key = self.endpoint_to_key(endpoint)
        self.store.put(key, json.dumps(result).encode())


class StartupRadarAPI:
    """
    Class to use the StartupRadar API.
    """

    PAGE_LIMIT_DEFAULT = 100
    MAX_PAGES_DEFAULT = 100

    def __init__(
        self,
        api_key: str,
        page_limit=PAGE_LIMIT_DEFAULT,
        max_pages=MAX_PAGES_DEFAULT,
        session_factory=None,
        cache: CacheI = None,
    ):
        self.api_key = api_key
        self.page_limit = page_limit
        self.max_pages = max_pages

        self.session_factory = requests.Session
        if session_factory:
            self.session_factory = session_factory

        self.cache = PassThroughCache()
        if cache:
            self.cache = cache

    @property
    def is_cached(self):
        return not isinstance(self.cache, PassThroughCache)

    def get_cached_or_none(self, endpoint):
        try:
            return self.cache.get(endpoint)
        except NotInCacheException:
            return None

    def _request(self, endpoint: str, params: dict = None):
        url = urljoin("https://api.startupradar.co/", endpoint)
        logging.debug(f"requesting endpoint ({url=}, {params=})")
        session = self.session_factory()
        response = session.get(url, params=params, headers={"X-ApiKey": self.api_key})

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 403:
            raise ForbiddenError(response.json()["detail"])
        elif response.status_code == 404:
            raise NotFoundError(
                f"resource not found ({url=}, response={response.json()})"
            )
        else:
            raise StartupRadarAPIError(
                "unhandled status code "
                f"({response.status_code}, {endpoint=}, {params=})"
            )

    def _request_with_cache(self, endpoint: str, params: dict = None):
        if params:
            logging.warning(
                f"cannot cache params, requesting un-cached ({endpoint=}, {params=})"
            )
            return self._request(endpoint, params)

        result_cache = self.get_cached_or_none(endpoint)
        # falsy is not sufficient -> empty results like lists
        if result_cache is not None:
            logging.debug(f"fetched from cache ({endpoint=})")
            result = result_cache
        else:
            logging.info(f"fetching uncached ({endpoint=})")
            result = self._request(endpoint)

            logging.debug(f"caching results ({endpoint=})")
            self.cache.put(endpoint, result)

        return result

    def _request_paged(self, endpoint: str, max_pages=MAX_PAGES_DEFAULT):
        result_cache = self.get_cached_or_none(endpoint)
        # falsy is not sufficient -> empty lists
        if result_cache is not None:
            logging.debug(f"fetched from cache ({endpoint=})")
            result = self.cache.get(endpoint)
        else:
            logging.info(f"fetching uncached {endpoint=}")
            pages = []
            for page in range(max_pages):
                response = self._request(
                    endpoint, {"page": page, "limit": self.page_limit}
                )

                # add results to pages
                pages.append(response)

                if len(response) < self.page_limit:
                    # less results than limit -> last page
                    break

            result = list(chain(*pages))

            logging.debug(f"caching result ({endpoint=})")
            self.cache.put(endpoint, result)

        return result

    def get(self):
        endpoint = "/"
        return self._request_with_cache(endpoint)

    def get_domain(self, domain: str):
        ensure_valid_domain(domain)

        endpoint = f"/web/domains/{domain}"
        return self._request_with_cache(endpoint)

    def generate_domains(self):
        has_results = True
        page = 0
        while has_results:
            response = self._request(
                "/web/domains", params={"page": page, "limit": self.page_limit}
            )
            for domain in response:
                yield domain

            has_results = len(response) > 0
            page += 1

    def get_text(self, domain: str):
        ensure_valid_domain(domain)

        endpoint = f"/web/domains/{domain}/text"
        return self._request_with_cache(endpoint)

    def get_links(self, domain: str):
        ensure_valid_domain(domain)

        endpoint = f"/web/domains/{domain}/links/domain-links"
        return self._request_paged(endpoint)

    def get_backlinks(self, domain: str):
        ensure_valid_domain(domain)

        if domain in DOMAINS_IGNORED_BACKLINKS:
            msg = (
                "domain is in ignored domains "
                "because it would return too many backlinks. "
                f"Returning empty backlinks instead ({domain=})"
            )
            logging.warning(msg)
            return []

        endpoint = f"/web/domains/{domain}/links/domain-backlinks"
        return self._request_paged(endpoint)

    def get_similar(self, domain: str):
        ensure_valid_domain(domain)

        endpoint = f"/web/domains/{domain}/similar"
        return self._request_with_cache(endpoint)

    def get_socials(self, domain: str):
        ensure_valid_domain(domain)
        endpoint = f"/web/domains/{domain}/socials"
        return self._request_with_cache(endpoint)

    def get_sources(self):
        return self._request_with_cache("/sources")


def ensure_valid_domain(domain: str):
    if not domain:
        raise InvalidDomainError(f"domain is falsy ({domain=})")

    if domain != domain.strip():
        raise InvalidDomainError(f"domain contains spaces ({domain=})")

    extraction = tldextract.extract("http://" + domain)
    domain_actual = extraction.registered_domain
    if domain_actual != domain:
        raise InvalidDomainError(f"domain is invalid ({domain=}, {domain_actual=})")


def is_valid_domain(domain: str):
    try:
        ensure_valid_domain(domain)
        return True
    except InvalidDomainError:
        return False


def get_text_or_empty_dict(api, domain) -> dict:
    try:
        return api.get_text(domain)
    except NotFoundError:
        return {}
