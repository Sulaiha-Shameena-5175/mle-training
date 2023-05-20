"""
The nonstandardcode contains the entire process.
The data is downloaded and split into test and train dataset.
Trained under different models.
The best parameters were identified through RandomizedSearchCV
"""


import logging
import os
import tarfile

import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.tree import DecisionTreeRegressor

logging.debug("--- The script start here ---")
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("app/datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
    It downloads the data as zip file and extracts it

    Parameters:
    -----------
    housing_url : string
        The url to download the data
    housing_path : string
        The path to store the data
    """
    logging.debug("--- fetch_housing_data starts here ---")
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    # urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    logging.debug("--- fetch_housing_data ends ---")


def load_housing_data(dataset_file_name, housing_path=HOUSING_PATH):
    # fetch_housing_data()
    """
    It loads the data and returns it as a dataframe

    Parameters:
    -----------
    dataset_file_name : string
        Name of the dataset
    housing_path : string
        The path to retrieve the data

    Return:
    -------
    dataframe : panda dataframe
    returns the dataframe of the data

    """
    csv_path = os.path.join(housing_path, dataset_file_name)
    return pd.read_csv(csv_path)


def get_housing_data(dataset_file_name):
    """
    It loads the data and returns it as a dataframe

    Parameters:
    -----------
    dataset_file_name : string
        Name of the dataset

    Return:
    -------
    dataframe : panda dataframe
    returns the dataframe of the data

    """
    housing = load_housing_data(dataset_file_name)
    housing["ocean_proximity"].replace(
        {
            "INLAND": 0,
            "<1H OCEAN": 1,
            "NEAR OCEAN": 2,
            "NEAR BAY": 3,
            "ISLAND": 4,
        },
        inplace=True,
    )
    print("housing", housing.head())
    return housing
