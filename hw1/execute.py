
import utilities as utl

import numpy as np
import itertools
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
    converted_table = [[labels_result[value - 1] for value in row] for row in table]
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
    """
    low traffic low-medium human - short greenlight
    low traffic high human - short greenlight
    high traffic - long greenlight
    medium traffic high human - medium greenlight
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


def compute_membership_value(x, j, k):
    """
    Calculate the membership function value for a given input value 'x' in a triangular fuzzy set.

    Parameters:
    x (numpy.ndarray or float): The input value or array of values.
    j (int): The index of the membership function within the range [1, k].
    k (int): The total number of membership functions in the set.

    Returns:
    membership (numpy.ndarray): The membership function values for the input 'x'.
    """
    a = (j - 1) / (k - 1)
    b = 1 / (k - 1)
    tmp_membership = 1 - np.abs(x - a) / b
    tmp_membership[tmp_membership < 0] = 0
    membership = tmp_membership
    return membership


def generate_combinations(k):
    """
    Generate all possible combinations for different k values using itertools.

    Args:
    k (list): A list specifying the number of fuzzy sets for each feature.

    Returns:
    combinations (list): A list of lists, where each inner list represents a combination of fuzzy set indices for
    each feature.
    num_combinations (int): An integer specifying the number of all generated combinations.
    """
    combinations = list(itertools.product(*[range(k_i) for k_i in k]))
    num_combinations = len(combinations)

    return combinations, num_combinations


def generate_partitions(k, min_val, max_val):
    """
    Generate an array with a 3-element list for each of the k partitions with values of a triangular shape function.

    Args:
    k (int): The number of partitions.
    min_val (int): The starting value of the range.
    max_val (int): The ending value of the range.

    Returns:
    spaces (np.ndarray): An array of shape (k, 3) where each row represents a triangular shape function with
    the format [beginning, peak value, ending] points.
    """

    # Ensure k is at least 2
    if k < 2:
        raise ValueError("k must be at least 2")

    # Create an array of shape (k, 3) to store the triangular shape function points
    spaces = np.zeros((k, 3))

    # Calculate the portion size
    portion = (max_val - min_val) / (k - 1)

    # Set the points for the first and last triangles
    spaces[0] = [min_val, min_val, min_val + portion]
    spaces[k - 1] = [max_val - portion, max_val, max_val]

    # Set the points for the middle triangles
    for i in range(1, k - 1):
        peak_value = min_val + i * portion
        spaces[i] = [peak_value - portion, peak_value, peak_value + portion]

    return spaces



def compute_membership_values_multidim(x, k):
    """
    Calculate the membership function values for a dataset 'x' with triangular fuzzy sets.

    Parameters:
    x (numpy.ndarray): The input dataset of shape (num_samples, num_features).
    k (numpy.ndarray): An array specifying the number of membership functions for each feature.

    Returns:
    membership_values (numpy.ndarray): A multidimensional array of membership function values for the input data 'x'.
                  The shape of the returned array is (num_samples, num_features, max(k)).
    """
    num_samples, num_features = x.shape
    membership_values = np.zeros((num_samples, num_features, np.max(k)))

    for i in range(num_features):
        feature_data = x[:, i]
        for j in range(np.max(k)):
            membership_values[:, i, j] = utl.compute_membership_value(feature_data, j + 1, k[i])

    return membership_values


def compute_degrees_of_compatibility(membership_values, k):
    """
    Compute the degree of compatibility for all possible combinations of fuzzy rule conditions.

    Args:
    membership_values (list): A list of membership values for each sample and feature.
    k (list): A list specifying the number of fuzzy sets for each feature.

    Returns:
    degree_of_compatibility (list): A list of lists, where each element represents the degree of compatibility for all
    combinations of conditions for each sample.
    """
    degree_of_compatibility = []

    combinations,_ = utl.generate_combinations(k)
    num_samples = len(membership_values)
    num_features = len(k)

    for i in range(num_samples):
        sample_combinations = []
        for combination in combinations:
            product_values = []
            for j in range(num_features):
                if len(membership_values[i]) == 1:
                    product_values.append(membership_values[i][0][combination[j]])
                else:
                    product_values.append(membership_values[i][j][combination[j]])
            product_values = np.array(product_values)
            sample_combinations.append(np.product(product_values))
        degree_of_compatibility.append(sample_combinations)

    return degree_of_compatibility


def compute_weights(degree_of_compatibility, alpha):
    """
    Compute weights from degrees of compatibility using a power transformation.

    Args:
    degree_of_compatibility (np.ndarray): An array of degrees of compatibility.
    alpha (float): A positive constant that controls the power transformation.

    Returns:
    weights (np.ndarray): An array of weights computed by raising the degrees of compatibility to the power of alpha.
    """
    weights = np.power(degree_of_compatibility, alpha)
    return weights


def compute_consequent_values(weights, output):
    """
    Compute the consequent values for different combinations of fuzzy rule conditions.

    Args:
    weights (numpy.ndarray): An array of weights for different rule combinations with
    shape [num_samples, num_combinations].
    output (numpy.ndarray): The true output values for each sample with shape [num_samples, 1].

    Returns:
    consequent_values (numpy.ndarray): An array of consequent values for each combination of fuzzy rule conditions
    with shape [num_combinations, 1].
    """
    num_samples, num_combinations = weights.shape

    numerator = np.zeros(num_combinations)
    denominator = np.zeros(num_combinations)

    for j in range(num_combinations):
        numerator[j] = np.dot(weights[:, j], output)
        denominator[j] = np.sum(weights[:, j])

    consequent_values = numerator / denominator

    consequent_values = consequent_values.reshape((num_combinations, 1))

    return consequent_values


def derive_linguistic_labels_and_predict(x, b, B, k):
    """
    Derive linguistic labels and predict output values using fuzzy inference.

    Args:
    x (numpy.ndarray): Input data for prediction with shape (num_samples, num_features).
    b (numpy.ndarray): Input data representing fuzzy rules with shape (num_rules, num_features).
    B (numpy.ndarray): Range of fuzzy sets for each feature with shape (num_features, num_fuzzy_sets).
    k (numpy.ndarray): Number of fuzzy sets for each feature with shape (num_features,).

    Returns:
    tuple: A tuple containing:
        - output_inferred (numpy.ndarray): Inferred output values with shape (num_samples,).
        - main_table (numpy.ndarray): Main linguistic labels for each combination of features with
        shape (num_combinations, num_features).
        - secondary_table (numpy.ndarray): Secondary linguistic labels for each combination of features with
        shape (num_combinations, num_features).
    """
    num_samples, num_features = x.shape
    combinations, num_combinations = utl.generate_combinations(k)

    # Calculate membership values
    membership_values = compute_membership_values_multidim(b, B)
    membership_values_x = compute_membership_values_multidim(x, k)

    # Calculate degrees of compatibility
    degree_of_compatibility = np.array(compute_degrees_of_compatibility(membership_values_x, k))

    # Calculate the main and secondary rule tables
    main_table = np.argmax(membership_values, axis=-1)
    sorted_indices = np.argsort(-membership_values, axis=-1)
    secondary_table = sorted_indices[:, :, 1]

    # Store the corresponding membership values
    membership_main = np.max(membership_values, axis=-1)
    membership_secondary = np.partition(membership_values, -2, axis=-1)[:, :, -2]

    # Create fuzzy sub-spaces
    space = utl.generate_partitions(B[0], 0, 1)

    numerator = np.zeros([num_samples, num_combinations])
    denominator = np.zeros([num_samples, num_combinations])
    for i in range(num_combinations):
        # Determine the center values of the fuzzy sub-spaces
        B1 = space[main_table[i]]
        B2 = space[secondary_table[i]]

        # Calculate numerator and denominator of the inferred output
        numerator[:, i] = (degree_of_compatibility[:, i] * B1[0][1] * membership_main[i] +
                           degree_of_compatibility[:, i] * B2[0][1] * membership_secondary[i])
        denominator[:, i] = (degree_of_compatibility[:, i] * membership_main[i] +
                             degree_of_compatibility[:, i] * membership_secondary[i])

    # Calculate inferred output
    output_inferred = np.sum(numerator, axis=1) / np.sum(denominator, axis=1)

    return output_inferred, main_table.reshape(k), secondary_table.reshape(k)


# Generate data based on these rules
num_data_points = 500
data = []

for _ in range(num_data_points):
    traffic_density = random.randint(0, 45)
    pedestrian_activity = random.randint(0, 50)
    traffic_light_timing = np.round(generate_traffic_light_timing(traffic_density, pedestrian_activity), 2)
    data.append([traffic_density, pedestrian_activity, traffic_light_timing])
data = np.array(data)

data_normalized = normalize_data(data)
input_data, output_data = data_normalized[:, 0:2], data_normalized[:, 2]

def execute(input_mat, output, k=np.array([5, 5]), alpha=5, B=np.array([5])):

    membership_values = compute_membership_values_multidim(input_mat, k)
    degree_of_compatibility = compute_degrees_of_compatibility(membership_values, k)
    weights = compute_weights(degree_of_compatibility, alpha)
    consequent_values = compute_consequent_values(weights, output)

    output_inferred, main_table, secondary_table = derive_linguistic_labels_and_predict(input_mat, consequent_values, B, k)

    # Compute Performance Index (PI)
    #pi = 0.5 * np.sum((output - output_inferred) ** 2)
    pi = (1/input_mat.shape[0]) * np.sum((output - output_inferred) ** 2)

    #print(f'K = {k}, alpha = {alpha}, PI = {pi}')
    return pi, main_table, secondary_table, output_inferred, consequent_values


k = np.array([3, 3])
alpha = 2
B = np.array([3])
pi, main_table, secondary_table, output_inferred, consequent_values = execute(input_mat=input_data, output=output_data, k=k, alpha=alpha, B=B)

print('\nMain rule table:')
print(main_table)
print('\nSecondary rule table:')
print(secondary_table)