import requests

"""
It contains the unit tests
"""


def checkUrlExist(url):
    response = requests.get(url)
    assert response.status_code < 400


checkUrlExist("https://www.google.com")
