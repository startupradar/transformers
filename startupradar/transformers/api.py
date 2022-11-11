"""
Classes to access the StartupRadar API.
"""

import logging
from urllib.parse import urljoin

import requests


class StartupRadarAPIError(RuntimeError):
    pass


class NotFoundError(StartupRadarAPIError):
    pass


class ForbiddenError(StartupRadarAPIError):
    pass


class StartupRadarAPI:
    """
    Class to use the StartupRadar API.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _request(self, endpoint: str, params: dict = None):
        url = urljoin("https://api.startupradar.co/", endpoint)
        logging.info(f"requesting endpoint ({url=}, {params=})")
        response = requests.get(url, params, headers={"X-ApiKey": self.api_key})
        if response.status_code == 200:
            return response
        elif response.status_code == 403:
            raise ForbiddenError(response.json()["detail"])
        else:
            raise StartupRadarAPIError(
                f"unhandled status code ({response.status_code})"
            )

    def _request_paged(self, endpoint: str):
        results = []
        for page in range(100):
            response = self._request(endpoint, {"page": page}).json()
            results.extend(response)
            if not response:
                break

        return results

    def get(self):
        return self._request("/").json()

    def get_text(self, domain: str):
        return self._request(f"/web/domains/{domain}/text").json()

    def get_links(self, domain):
        endpoint = f"/web/domains/{domain}/links/domain-links"
        return self._request_paged(endpoint)

    def get_backlinks(self, domain: str):
        endpoint = f"/web/domains/{domain}/links/domain-backlinks"
        return self._request_paged(endpoint)
