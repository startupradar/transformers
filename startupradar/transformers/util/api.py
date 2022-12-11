"""
Classes to access the StartupRadar API.
"""
import calendar
import logging
from datetime import datetime, timedelta
from email.utils import parsedate, formatdate, parsedate_to_datetime
from itertools import chain
from urllib.parse import urljoin

import cachecontrol
import requests
import tldextract
from cachecontrol import CacheController
from cachecontrol.caches import FileCache
from cachecontrol.heuristics import BaseHeuristic
from requests.adapters import HTTPAdapter

from startupradar.transformers.util.exceptions import (
    StartupRadarAPIError,
    NotFoundError,
    ForbiddenError,
    InvalidDomainError,
)

DOMAINS_IGNORED_BACKLINKS = (
    "facebook.com",
    "twitter.com",
    "instagram.com",
    "linkedin.com",
)


class OneWeekHeuristic(BaseHeuristic):
    TIMEDELTA = timedelta(weeks=1)

    def update_headers(self, response):
        date = parsedate(response.headers["date"])
        expires = datetime(*date[:6]) + self.TIMEDELTA
        return {
            "expires": formatdate(calendar.timegm(expires.timetuple())),
            "cache-control": "public",
        }

    def warning(self, response):
        msg = "Automatically cached! Response is Stale."
        return '110 - "%s"' % msg


class CachesNotFoundController(CacheController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # hack to also cache 404s without changing anything else
        if 404 not in self.cacheable_status_codes:
            self.cacheable_status_codes = self.cacheable_status_codes + (404,)


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
    ):
        self.api_key = api_key
        self.page_limit = page_limit
        self.max_pages = max_pages

        cache_control = cachecontrol.CacheControl(
            requests.Session(),
            cache=FileCache(".cachecontrol"),
            heuristic=OneWeekHeuristic(),
            controller_class=CachesNotFoundController,
        )
        self.session_factory = lambda: cache_control
        if session_factory:
            self.session_factory = session_factory

    @property
    def is_cached(self):
        adapter = self.session_factory().get_adapter("https://")
        is_cached_adapter = isinstance(
            adapter, cachecontrol.adapter.CacheControlAdapter
        )
        is_regular_adapter = isinstance(adapter, HTTPAdapter)
        if not is_cached_adapter and not is_regular_adapter:
            logging.warning(
                "unknown cache adapter used, "
                f"caching detection not working ({adapter=})"
            )

        return is_cached_adapter

    def _request(self, endpoint: str, params: dict = None):
        url = urljoin("https://api.startupradar.co/", endpoint)
        logging.debug(f"requesting endpoint ({url=}, {params=})")
        session = self.session_factory()
        response = session.get(url, params=params, headers={"X-ApiKey": self.api_key})

        # hack to check if it's a cache miss, i.e. newly fetched
        if "expires" in response.headers:
            expires_date = parsedate_to_datetime(response.headers["expires"])
            cache_date = expires_date - OneWeekHeuristic.TIMEDELTA
            age = datetime.utcnow() - cache_date
            if age < timedelta(seconds=10):
                logging.info(f"got fresh response ({url=} {params=})")

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

    def _request_paged(self, endpoint: str, max_pages=MAX_PAGES_DEFAULT):
        pages = []
        for page in range(max_pages):
            response = self._request(endpoint, {"page": page, "limit": self.page_limit})

            # add results to pages
            pages.append(response)

            if len(response) < self.page_limit:
                # less results than limit -> last page
                break

        return list(chain(*pages))

    def get(self):
        endpoint = "/"
        return self._request(endpoint)

    def get_domain(self, domain: str):
        self._ensure_valid_domain(domain)

        endpoint = f"/web/domains/{domain}"
        return self._request(endpoint)

    def get_text(self, domain: str):
        self._ensure_valid_domain(domain)

        endpoint = f"/web/domains/{domain}/text"
        return self._request(endpoint)

    def get_links(self, domain: str):
        self._ensure_valid_domain(domain)

        endpoint = f"/web/domains/{domain}/links/domain-links"
        return self._request_paged(endpoint)

    def get_backlinks(self, domain: str):
        self._ensure_valid_domain(domain)

        if domain in DOMAINS_IGNORED_BACKLINKS:
            msg = (
                "domain is in ignored domains "
                "because it would return too many backlinks"
                "returning empty backlinks instead {domain=}"
            )
            logging.warning(msg)
            return []

        endpoint = f"/web/domains/{domain}/links/domain-backlinks"
        return self._request_paged(endpoint)

    def get_similar(self, domain: str):
        self._ensure_valid_domain(domain)

        endpoint = f"/web/domains/{domain}/similar"
        return self._request(endpoint)

    def get_socials(self, domain: str):
        self._ensure_valid_domain(domain)
        endpoint = f"/web/domains/{domain}/socials"
        return self._request(endpoint)

    def get_sources(self):
        return self._request("/sources")

    def _ensure_valid_domain(self, domain: str):
        if not domain:
            raise InvalidDomainError(f"domain is falsy ({domain=})")

        if domain != domain.strip():
            raise InvalidDomainError(f"domain contains spaces ({domain=})")

        extraction = tldextract.extract("http://" + domain)
        domain_actual = extraction.registered_domain
        if domain_actual != domain:
            raise InvalidDomainError(f"domain is invalid ({domain=}, {domain_actual=})")
