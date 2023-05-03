import logging
import os
import pickle
import tarfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import randint
from six.moves import urllib
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


def train_train_split_data(housing):
    """
    It splits the data into test and train subsets

    Parameters:
    -----------
    housing : dataframe
        The data that need to be split

    Return:
    -------
    returns
        housing_prepared,
        housing_labels,
        strat_train_set,
        strat_test_set,
        imputer

    """
    train_set, test_set = train_test_split(
        housing, test_size=0.2, random_state=42
    )

    logging.debug("test train split done")

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    train_set, test_set = train_test_split(
        housing, test_size=0.2, random_state=42
    )

    compare_props = pd.DataFrame(
        {
            "Overall": income_cat_proportions(housing),
            "Stratified": income_cat_proportions(strat_test_set),
            "Random": income_cat_proportions(test_set),
        }
    ).sort_index()
    compare_props["Rand. %error"] = (
        100 * compare_props["Random"] / compare_props["Overall"] - 100
    )
    compare_props["Strat. %error"] = (
        100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    )

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.copy()
    housing.plot(kind="scatter", x="longitude", y="latitude")
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

    corr_matrix = housing.corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)
    housing["rooms_per_household"] = (
        housing["total_rooms"] / housing["households"]
    )
    housing["bedrooms_per_room"] = (
        housing["total_bedrooms"] / housing["total_rooms"]
    )
    housing["population_per_household"] = (
        housing["population"] / housing["households"]
    )

    housing = strat_train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()

    imputer = SimpleImputer(strategy="median")

    housing_num = housing.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(
        X, columns=housing_num.columns, index=housing.index
    )
    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(
        pd.get_dummies(housing_cat, drop_first=True)
    )

    logging.debug("preforming Linear Regression")
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)
    create_pickle_file(lin_reg, "lin_reg.pkl")

    housing_predictions = lin_reg.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    lin_rmse

    lin_mae = mean_absolute_error(housing_labels, housing_predictions)
    lin_mae

    logging.debug("preforming Decision Tree Regressor")
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)
    create_pickle_file(tree_reg, "tree_reg.pkl")

    housing_predictions = tree_reg.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    tree_rmse
    return (
        housing_prepared,
        housing_labels,
        strat_train_set,
        strat_test_set,
        imputer,
    )


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)


def create_pickle_file(model, fileName):
    """
    It pickles the model

    Parameters:
    -----------
    model : ml model
        The ml model that needs to be pickled
    fileName: string
        Name of the pickle file
    """
    pickle.dump(model, open("../artifacts/.gitkeep/" + fileName, "wb"))
    logging.debug(f"pickle file created {fileName}")
