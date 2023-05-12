import codecs
import os

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = "0.0.13"
DESCRIPTION = "ML project that predicts the house price"

# Setting up
setup(
    name="MLE_Training_5175",
    version=VERSION,
    author="MLE",
    author_email="sulaiha.shameena@tigeranalytics.com",
    description=DESCRIPTION,
    packages=find_packages(),
    keywords=["python", "mle"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)
