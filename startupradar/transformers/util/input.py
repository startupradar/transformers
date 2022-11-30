import logging
import re
from enum import Enum

import requests
from tldextract import tldextract

from startupradar.transformers.util.api import StartupRadarAPI


class RedirectHandling(Enum):
    NONE = None
    FOLLOW = "follow"
    FILTER = "filter"


class ErrorHandling(Enum):
    IGNORE = "ignore"
    WARN = "warn"
    RAISE = None


class DomainInputCleaner:
    """
    Takes raw input and transforms it into clean domains.
    """

    api = None
    error_handling = None
    redirect_handling = None
    fix_unclean = True

    def __init__(
        self,
        api: StartupRadarAPI,
        error_handling: ErrorHandling = None,
        redirect_handling: RedirectHandling = None,
        fix_unclean: bool = True,
    ):
        self.api = api

        self.error_handling = ErrorHandling(error_handling)
        self.redirect_handling = RedirectHandling(redirect_handling)
        self.fix_unclean = fix_unclean

    def get(self, urls):
        # extract domains
        domain_gen = self._generate_domains_from_urls(urls)

        # get redirects
        domains = []
        for domain in domain_gen:
            if self.redirect_handling == RedirectHandling.FOLLOW:
                domains.append(get_response_domain(domain))
            elif self.redirect_handling == RedirectHandling.FILTER:
                response_domain = get_response_domain(domain)
                if response_domain == domain:
                    domains.append(domain)
                else:
                    logging.info(
                        "Filtered domain due to redirect "
                        f"({domain=}, {response_domain=})"
                    )
            else:
                # just add
                domains.append(domain)

        return domains

    def get_unique(self, urls) -> set:
        return set(self.get(urls))

    def _generate_domains_from_urls(self, urls):
        for url in urls:
            try:
                yield self._url_to_domain(url)
            except RuntimeError:
                if self.error_handling == ErrorHandling.WARN:
                    logging.warning(f"Failed to extract domain for {url=}")
                elif self.error_handling == ErrorHandling.RAISE:
                    raise

    def _url_to_domain(self, url) -> str:
        if self.fix_unclean:
            if not re.match(r"^https?://.*", url):
                url = "http://" + url

        domain = tldextract.extract(url).registered_domain
        if not domain:
            raise RuntimeError(f"not a domain {url=}")

        return domain


def get_response_domain(domain) -> str:
    previous_url = None
    current_url = "http://" + domain
    while previous_url != current_url:
        resp = requests.head(current_url, allow_redirects=True)
        previous_url, current_url = current_url, resp.url
    response_domain = tldextract.extract(resp.url).registered_domain
    return response_domain
