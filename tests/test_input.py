import pytest

from startupradar.transformers.util.input import DomainInputCleaner, RedirectHandling


@pytest.fixture
def api():
    # not used currently, so none is fine
    return None


def test_domain_input_cleaner(api):
    dic = DomainInputCleaner(api)
    result = dic.get(["https://karllorey.com"])
    assert result == ["karllorey.com"]


def test_domain_input_cleaner_raw_domain(api):
    dic = DomainInputCleaner(api)
    result = dic.get(["karllorey.com"])
    assert result == ["karllorey.com"]


@pytest.mark.vcr
def test_domain_input_cleaner_redirect(api):
    url = "http://pointnine.vc"

    dic = DomainInputCleaner(api, redirect_handling=RedirectHandling.FOLLOW)
    print(dic.redirect_handling)
    result = dic.get([url])
    assert result == ["pointninecap.com"]
