import requests

SAMPLE_DATA_URL = "http://localhost:8000/NeuLogAPI?start=0&end=60"


def get_experiment_data():
    response = requests.get(url=SAMPLE_DATA_URL)
    if response.ok:
        print(response.json())
    else:
        print("Cannot get Data")
        return False

get_experiment_data()
