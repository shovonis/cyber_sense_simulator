import pandas as pd


def data_to_numeric(data):
    data.HR = pd.to_numeric(data.HR)
    data.GSR = pd.to_numeric(data.GSR)
    return data


def calculate_min_max(data, window=3):
    # Calculate Min
    data['HR_MIN'] = data['HR'].rolling(window=window).min()
    data['GSR_MIN'] = data['GSR'].rolling(window=window).min()

    # Calculate Max
    data['HR_MAX'] = data['HR'].rolling(window=window).max()
    data['GSR_MAX'] = data['GSR'].rolling(window=window).max()

    return data


def calculate_percentage_of_change(data, resting_data):
    # Calculate difference from Resting individual_raw_data
    data['PC_HR'] = (data['HR'] - resting_data['HR'].mean()) / resting_data['HR'].mean()
    data['PC_GSR'] = (data['GSR'] - resting_data['GSR'].mean()) / resting_data['GSR'].mean()

    # calculate percentage of change for each individual_raw_data
    data['PC_HR'] = data['PC_HR'] * 100.0
    data['PC_GSR'] = data['PC_GSR'] * 100.0

    return data


def calculate_moving_avg(data, window=3):
    data['HR'] = data['HR'].rolling(window=window).mean()
    data['GSR'] = data['GSR'].rolling(window=window).mean()

    return data


