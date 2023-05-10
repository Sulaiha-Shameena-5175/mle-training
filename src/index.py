"""
The nonstandardcode contains the entire process.
The data is downloaded and split into test and train dataset.
Trained under different models.
The best parameters were identified through RandomizedSearchCV
"""


import argparse
import logging

from MLE_Training_Package import ingest_data, score, train

parser = argparse.ArgumentParser()
parser.add_argument("dataset")
args = parser.parse_args()
print("dataset --> " + args.dataset)
dataset_file_name = args.dataset

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    filename="../logs/.gitkeep/logfile.log",
    filemode="w",
    format="%(name)s-%(levelname)s-%(message)s",
    level=logging.DEBUG,
    force=True,
)

logging.debug("Loading the dataset")
# ingest_data.success()
housing = ingest_data.get_housing_data(dataset_file_name)  # "housing.csv"

logging.debug("split the dataset into test and train")
(
    housing_prepared,
    housing_labels,
    strat_train_set,
    strat_test_set,
    imputer,
) = train.train_train_split_data(housing)

logging.debug("identify the best paramaters using RandomizedSearchCV")
score_calculated = score.calc_score(
    housing_prepared, housing_labels, strat_train_set, strat_test_set, imputer
)
