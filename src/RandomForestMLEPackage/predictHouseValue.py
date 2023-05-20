import pickle


def load_data_from_pickle(filepath):
    with open(filepath, "rb") as pickle_file:
        content = pickle.load(pickle_file)
        return content


def predictHousing(filelocation, data):
    rf = load_data_from_pickle(filelocation)
    result = rf.predict(data)
    print(result)


"""
predictHousing(
    "random.pkl",
    [
        [
            0.59229422,
            -0.71065803,
            0.02756357,
            1.78850799,
            1.16374818,
            0.68509554,
            1.23238474,
            2.31286606,
            0.48828718,
            -0.07091122,
            -0.86813563,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ]
    ],
)
"""
