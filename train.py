import os
import pickle
import sys
import warnings

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


"""
def load_data(data_path):
    data = pd.read_csv(data_path)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]
    return train_x, train_y, test_x, test_y
"""


def load_data_from_pickle(filepath):
    with open(filepath, "rb") as pickle_file:
        content = pickle.load(pickle_file)
        return content


def get_data():
    housing_prepared = load_data_from_pickle(
        "artifacts/.gitkeep/housing_prepared.pkl"
    )
    housing_labels = load_data_from_pickle(
        "artifacts/.gitkeep/housing_labels.pkl"
    )
    x_test = load_data_from_pickle("artifacts/.gitkeep/x_test.pkl")
    y_test = load_data_from_pickle("artifacts/.gitkeep/y_test.pkl")
    return housing_prepared, housing_labels, x_test, y_test


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    # data_path = "data/raw/.gitkeep/wine-quality.csv"
    # train_x, train_y, test_x, test_y = load_data(data_path)
    housing_prepared, housing_labels, x_test, y_test = get_data()

    n_estimators = float(sys.argv[1]) if len(sys.argv) > 1 else 3
    max_features = float(sys.argv[2]) if len(sys.argv) > 2 else 3
    n_estimators = 10
    max_features = 3
    experiment_id = mlflow.create_experiment("experiment_rf")
    with mlflow.start_run(
        run_name="PARENT_RUN",
        experiment_id=experiment_id,
        tags={"version": "v1", "priority": "P1"},
        description="parent",
    ) as parent_run:
        mlflow.log_param("child", "yes")
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            random_state=42,
        )
        rf.fit(housing_prepared, housing_labels)

        predicted_qualities = rf.predict(x_test)

        (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)

        print(
            "RandomForest model (n_estimators=%f, max_features=%f):"
            % (n_estimators, max_features)
        )
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_features", max_features)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(rf, "model")
