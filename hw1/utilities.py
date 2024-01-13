import numpy as np
import pandas as pd
import random


def table_to_linguistic_dataframe(table,
                                  labels_x1,
                                  labels_x2,
                                  labels_result):
    """
    convert 2d rule table into linguistic, assume tabel has the same size as label
    assume first dimension is x1, second dimension is x2
    Parameters:
    table (List[List[int]]): numeric fuzzy result, assume all list elements of the table has the same length
    labels_x1 (List[str]): list of labels for variable x1, length must match length of table
    labels_x2 (List[str]): list of labels for variable x2, length must match length of an element of table
    labels_result (List[str]): list of labels for resulting fuzzy label, index must match the value-1 of the table

    Returns:
    converted_table (pd.DataFrame): pandas dataframe with linguistic label index and column and convert values to appropriate linguistic label
    """
    assert len(table) > 0, "length of table must be more than 0!"
    assert len(table) == len(
        labels_x1), "length of the table ({}) is not equal to length of the labels for x1({})!".format(len(table),
                                                                                                       len(labels_x1))
    assert len(table[0]) == len(
        labels_x2), "length of the table ({}) is not equal to length of the labels for x2({})!".format(len(table),
                                                                                                       len(labels_x2))
    # assume that the 1 is the lowest value
    converted_table = [[labels_result[value] for value in row] for row in table]
    dataframe = pd.DataFrame(converted_table, columns=labels_x2, index=labels_x1)
    return dataframe


def generate_traffic_light_timing(traffic_density, pedestrian_activity):
    """
    Generate traffic light timing rule in seconds given numerical traffic density and pedestrian activity
    Fuzzy green traffic light timing in seconds:
    short 20-30,
    medium 30-45,
    long 45-60

    assumption of traffic is as follows, 3 lanes of cars.
    traffic density rule:
    0-12 (about 0-6 rows of cars) - low
    12-30 (about 4-10 rows of cars) - medium
    30-45 (about 7-15 rows of cars) - high

    for humans, assume not so dense human area
    0-15 - low
    15-35 - medium
    35-50 - high

    Parameters:
    traffic_density (int): given traffic density in terms of cars per minute, range of 0-45
    pedestrian_activity (int): given pedestrian activity in terms of humans per minute at intersection, range of 0-50

    Returns:
    greenlight time (int):
    """
    if traffic_density <= 12 and pedestrian_activity <= 15:
        return random.uniform(30, 45)
    elif traffic_density <= 12 and 15 < pedestrian_activity <= 35:
        return random.uniform(20, 30)
    elif traffic_density <= 12 and pedestrian_activity > 35:
        return random.uniform(20, 30)
    elif 12 < traffic_density <= 30 and pedestrian_activity <= 15:
       return random.uniform(45, 60)
    elif 12 < traffic_density <= 30 and 15 < pedestrian_activity <= 35:
        return random.uniform(45, 60)
    elif 12 < traffic_density <= 30 and pedestrian_activity > 35:
        return random.uniform(30, 45)
    elif 30 < traffic_density and pedestrian_activity <= 15:
        return random.uniform(45, 60)
    elif 30 < traffic_density and 15 < pedestrian_activity <= 35:
        return random.uniform(45, 60)
    elif 30 < traffic_density and pedestrian_activity > 35:
        return random.uniform(30, 45)
    return random.uniform(20, 60)


def normalize_data(data, feature_range=(0, 1)):
    """
    Perform Min-Max normalization on a dataset.

    Parameters:
    data (numpy.ndarray): The input dataset to be normalized.
    feature_range (tuple): A tuple specifying the desired output range (min, max). Default is (0, 1).

    Returns:
    normalized_data (numpy.ndarray): The normalized dataset.
    """
    # Calculate the minimum and maximum values for each feature
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)

    # Define the range for normalization
    min_range, max_range = feature_range

    # Perform Min-Max normalization
    normalized_data = (data - min_val) / (max_val - min_val) * (max_range - min_range) + min_range

    return normalized_data


def denormalize_data(normalized_data, min_val, max_val, feature_range=(0, 1)):
    """
    Denormalize a dataset that was previously normalized using Min-Max scaling.

    Parameters:
    normalized_data (numpy.ndarray): The normalized dataset.
    min_val (numpy.ndarray): The original minimum values for each feature.
    max_val (numpy.ndarray): The original maximum values for each feature.
    feature_range (tuple): A tuple specifying the desired output range (min, max). Default is (0, 1).

    Returns:
    denormalized_data (numpy.ndarray): The denormalized dataset.
    """
    min_range, max_range = feature_range
    denormalized_data = (normalized_data - min_range) / (max_range - min_range) * (max_val - min_val) + min_val
    return denormalized_data
