"""
Classes to access the StartupRadar API.
"""
import json
import logging
from abc import ABC
from datetime import datetime
from enum import Enum
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


class ResponseStatus(Enum):
    OK = "ok"
    NOT_FOUND = "not found"


class CachedResponse:
    status: ResponseStatus = None
    data = None

    def __init__(self, status, data=None):
        self.status = ResponseStatus(status)
        self.data = data

    def to_json(self):
        return {"status": self.status.value, "data": self.data}

    def to_bytes(self) -> bytes:
        return json.dumps(self.to_json()).encode()

    @staticmethod
    def from_bytes(bytes_: bytes):
        return CachedResponse(**json.loads(bytes_))

    def is_not_found(self):
        return self.status == ResponseStatus.NOT_FOUND

    def __repr__(self):
        return f"CachedResponse({self.status=})"


class CacheI(ABC):
    def get(self, endpoint: str) -> CachedResponse:
        raise NotImplementedError()

    def put(self, endpoint: str, result: CachedResponse):
        raise NotImplementedError()


class PassThroughCache(CacheI):
    def get(self, endpoint: str) -> CachedResponse:
        raise NotInCacheException()

    def put(self, endpoint: str, result: CachedResponse):
        return


class MinimalKeyValueCache(CacheI):
    def __init__(self, store: KeyValueStore):
        self.store = store

    def endpoint_to_key(self, endpoint: str):
        return quote(endpoint, safe="")

    def get(self, endpoint: str) -> CachedResponse:
        try:
            key = self.endpoint_to_key(endpoint)
            cache_bytes = self.store.get(key)
            return CachedResponse.from_bytes(cache_bytes)
        except KeyError:
            raise NotInCacheException from KeyError

    def put(self, endpoint: str, result: CachedResponse):
        key = self.endpoint_to_key(endpoint)
        self.store.put(key, result.to_bytes())


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

    def _request(self, endpoint: str, params: dict = None):
        """
        request a single endpoint/page.
        """
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

    def _request_paged(self, endpoint, max_pages=MAX_PAGES_DEFAULT):
        """
        request and return all pages of an endpoint.
        """
        pages = []
        for page in range(max_pages):
            response = self._request(endpoint, {"page": page, "limit": self.page_limit})

            # add results to pages
            pages.append(response)

            # fewer results than limit -> last page
            if len(response) < self.page_limit:
                break

        result = list(chain(*pages))
        return result

    def _request_with_cache(self, endpoint: str, params: dict = None):
        # without cache if params exist
        if params:
            logging.warning(
                f"cannot cache params, requesting un-cached ({endpoint=}, {params=})"
            )
            return self._request(endpoint, params)

        fetcher = lambda: self._request(endpoint)
        return self._cached_or_fetched(endpoint, fetcher)

    def _request_paged_with_cache(self, endpoint: str, max_pages=MAX_PAGES_DEFAULT):
        fetcher = lambda: self._request_paged(endpoint, max_pages)
        return self._cached_or_fetched(endpoint, fetcher)

    def _cached_or_fetched(self, endpoint, result_fetcher):
        """
        returns a cached result or fetches the endpoint.
        """

        try:
            # try to load from cache
            cached_response = self.cache.get(endpoint)
            logging.debug(f"fetched from cache ({endpoint=})")

            # raise error to mimic real api
            if cached_response.is_not_found():
                raise NotFoundError()
        except NotInCacheException:
            # do a real request
            try:
                logging.info(f"fetching uncached ({endpoint=})")
                result = result_fetcher()
                cached_response = CachedResponse(ResponseStatus.OK, result)
            except NotFoundError:
                # real request resulted in 404
                # -> store 404
                logging.debug(f"got 404, storing not found in cache ({endpoint=})")
                cached_response = CachedResponse(ResponseStatus.NOT_FOUND)

            logging.debug(f"caching result ({endpoint=}, {cached_response=})")
            self.cache.put(endpoint, cached_response)

        return cached_response.data

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
        return self._request_paged_with_cache(endpoint)

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
        return self._request_paged_with_cache(endpoint)

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

    def get_whois(self, domain: str):
        ensure_valid_domain(domain)
        endpoint = f"/web/domains/{domain}/whois"
        resp_raw = self._request_with_cache(endpoint)
        return {
            k: parse_date_or_none(resp_raw[k])
            for k in ["created", "changed", "expires"]
        }


def parse_date_or_none(raw: str):
    if not raw:
        return None

    return datetime.strptime(raw, "%Y-%m-%d")


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
