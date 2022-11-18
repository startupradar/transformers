"""
Classes to access the StartupRadar API.
"""

import logging
import pickle
from pathlib import Path
from urllib.parse import urljoin, quote

import requests


class StartupRadarAPIError(RuntimeError):
    pass


class NotFoundError(StartupRadarAPIError):
    pass


class ForbiddenError(StartupRadarAPIError):
    pass


class NotInCacheError(RuntimeError):
    pass


class APICache:
    """
    Plain caching.
    """

    def __init__(self, path=".cache"):
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key) -> Path:
        return self._path.joinpath(quote(key, safe=""))

    def has(self, key):
        return self._key_to_path(key).is_file()

    def get(self, key):
        path = self._key_to_path(key)
        if not path.is_file():
            raise NotInCacheError()
        try:
            with path.open("rb") as file:
                return pickle.load(file)
        except Exception as e:
            raise RuntimeError(f"getting cache key failed ({key=})") from e

    def put(self, key, value):
        path = self._key_to_path(key)
        with path.open("wb") as file:
            pickle.dump(value, file)


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

        self.session_factory = requests.session
        if session_factory:
            self.session_factory = session_factory

        self.cache = APICache()

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

    def _request_cached(self, endpoint, params=None):
        assert params is None, "cannot cache params yet"

        if self.cache.has(endpoint):
            result = self.cache.get(endpoint)
        else:
            result = self._request(endpoint)
            self.cache.put(endpoint, result)
        return result

    def _request_paged_cached(self, endpoint, params=None):
        assert params is None, "cannot cache params yet"

        try:
            # early return
            return self.cache.get(endpoint)
        except NotInCacheError:
            pass
        except Exception:
            logging.exception("reading cache failed, re-fetching...")

        # do not use cached responses here,
        # this will lead to mis-matching pages,
        # e.g. if runs are aborted on page 5/10
        result = self._request_paged(endpoint)
        self.cache.put(endpoint, result)

        return result

    def get_domain(self, domain: str):
        endpoint = f"/web/domains/{domain}"
        return self._request_cached(endpoint)

    def get_text(self, domain: str):
        endpoint = f"/web/domains/{domain}/text"
        return self._request_cached(endpoint)

    def get_links(self, domain: str):
        endpoint = f"/web/domains/{domain}/links/domain-links"
        return self._request_paged_cached(endpoint)

    def get_backlinks(self, domain: str):
        endpoint = f"/web/domains/{domain}/links/domain-backlinks"
        return self._request_paged_cached(endpoint)
