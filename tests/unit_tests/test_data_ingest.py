import pytest
import requests

"""
It contains the unit tests
"""


@pytest.mark.parametrize(
    "url",
    [
        ("https://raw.githubusercontent.com/ageron/handson-ml/master/"),
        ("https://www.google.com"),
    ],
)
def test_checkUrlExist(url):
    response = requests.get(url)
    assert response.status_code < 400


def test_always_passes():
    assert True


def test_always_fails():
    assert False
