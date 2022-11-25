"""
Classes to access the StartupRadar API.
"""
import calendar
import logging
from datetime import datetime, timedelta
from email.utils import parsedate, formatdate
from urllib.parse import urljoin

import cachecontrol
import requests
import tldextract
from cachecontrol.caches import FileCache
from cachecontrol.heuristics import BaseHeuristic

DOMAINS_IGNORED_BACKLINKS = (
    "facebook.com",
    "twitter.com",
    "instagram.com",
    "linkedin.com",
)


class StartupRadarAPIError(RuntimeError):
    pass


class NotFoundError(StartupRadarAPIError):
    pass


class ForbiddenError(StartupRadarAPIError):
    pass


class StartupRadarAPIWrapperError(RuntimeError):
    pass


class InvalidDomainError(StartupRadarAPIWrapperError):
    pass


class OneWeekHeuristic(BaseHeuristic):
    def update_headers(self, response):
        date = parsedate(response.headers["date"])
        expires = datetime(*date[:6]) + timedelta(weeks=1)
        return {
            "expires": formatdate(calendar.timegm(expires.timetuple())),
            "cache-control": "public",
        }

    def warning(self, response):
        msg = "Automatically cached! Response is Stale."
        return '110 - "%s"' % msg


class StartupRadarAPI:
    """
    Class to use the StartupRadar API.
    """

    PAGE_LIMIT_DEFAULT = 100

    def __init__(
        self, api_key: str, page_limit=PAGE_LIMIT_DEFAULT, session_factory=None
    ):
        self.api_key = api_key
        self.page_limit = page_limit

        cache_control = cachecontrol.CacheControl(
            requests.Session(),
            cache=FileCache(".cachecontrol"),
            heuristic=OneWeekHeuristic(),
        )
        self.session_factory = lambda: cache_control
        if session_factory:
            self.session_factory = session_factory

    def _request(self, endpoint: str, params: dict = None):
        url = urljoin("https://api.startupradar.co/", endpoint)
        logging.info(f"requesting endpoint ({url=}, {params=})")
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

    def _request_paged(self, endpoint: str):
        results = []
        for page in range(100):
            response = self._request(endpoint, {"page": page, "limit": self.page_limit})
            results.extend(response)
            if not response:
                break

        return results

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

    def _ensure_valid_domain(self, domain: str):
        if not domain:
            raise InvalidDomainError(f"domain is falsy ({domain=})")

        if domain != domain.strip():
            raise InvalidDomainError(f"domain contains spaces ({domain=})")

        extraction = tldextract.extract("http://" + domain)
        domain_actual = extraction.registered_domain
        if domain_actual != domain:
            raise InvalidDomainError(f"domain is invalid ({domain=}, {domain_actual=})")
