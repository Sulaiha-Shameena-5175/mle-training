import importlib
import sys

import pytest


@pytest.fixture(params=[""])
def printFixtures(req):
    print(req.param)


@pytest.mark.parametrize("name", [("pandas"), ("abc"), ("axe")])
def test_package_exist(name):
    try:
        module = importlib.import_module(name)
        print(module)
        assert True
    except Exception as e:
        print(e)
        assert False
