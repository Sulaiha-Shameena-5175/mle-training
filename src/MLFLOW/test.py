import json

import requests

# curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["alcohol", "chlorides", "citric acid", "density", "fixed acidity", "free sulfur dioxide", "pH", "residual sugar", "sulphates", "total sulfur dioxide", "volatile acidity"],"data":[[12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]]}' http://0.0.0.0:1234/invocations

url = "http://0.0.0.0:1234/invocations"
headers = {"Content-Type": "application/json"}

request_data = json.dumps(
    {
        "dataframe_split": {
            "data": [[-121.46, 38.52, 29.0, 3873.0, 797.0, 2237.0, 706.0, 0]],
        }
    }
)
response = requests.post(url, request_data, headers=headers)
print(response.text)
