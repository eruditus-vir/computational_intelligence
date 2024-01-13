import itertools

import numpy as np


class FuzzyInference:
    def __init__(self,
                 input_mat,
                 output_vec):
        """
        Parameters:
        input_mat (numpy.ndarray): Input data for prediction with shape (num_samples, num_features).
        output_vec (numpy.ndarray): Output data (num_samples, ).
        """
        self.input_mat = input_mat
        self.output_vec = output_vec

    @staticmethod
    def compute_membership_value(x, j, k):
        """
        Calculate the membership function value for a given input value 'x' in a triangular fuzzy set.

        Parameters:
        x (numpy.ndarray or float): The input value or array of values of dim (number of sample, 1)
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def compute_weights(degree_of_compatibility, alpha):
        """
        Compute weights from degrees of compatibility using a power transformation.

        Args:
        degree_of_compatibility (np.ndarray): An array of degrees of compatibility.
        alpha (float): A positive constant that controls the power transformation.

        Returns:
        weights (np.ndarray): An array of weights computed by raising the degrees of compatibility to the power of alpha.
        """
        return np.power(degree_of_compatibility, alpha)

    def compute_membership_values_multidim(self, x, k):
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
        max_member = np.max(k)
        membership_values = np.zeros((num_samples, num_features, max_member))

        for i in range(num_features):
            for j in range(max_member):
                membership_values[:, i, j] = self.compute_membership_value(x[:, i], j + 1, k[i])

        return membership_values

    def compute_degrees_of_compatibility(self, membership_values, k):
        """
        Compute the degree of compatibility for all possible combinations of fuzzy rule conditions.

        Args:
        membership_values (list): A list of membership values for each sample and feature.

        Returns:
        degree_of_compatibility (list): A list of lists, where each element represents the degree of compatibility for all
        combinations of conditions for each sample.
        """

        def generate_product_values_per_combination(si, comb):
            return np.array([membership_values[si][0][comb[j]] if len(membership_values[si]) == 1 \
                                 else membership_values[si][j][comb[j]] for j in range(len(k))])

        degree_of_compatibility = []
        combinations = self.generate_combinations(k)[0]
        for i in range(len(membership_values)):
            sample_combinations = [np.product(
                generate_product_values_per_combination(i, combination)) for combination in combinations]
            degree_of_compatibility.append(sample_combinations)
        return degree_of_compatibility

    @staticmethod
    def compute_consequent_values(weights, output_data):
        """
        Compute the consequent values for different combinations of fuzzy rule conditions.

        Args:
        weights (numpy.ndarray): An array of weights for different rule combinations with
        shape [num_samples, num_combinations].
        output_data (numpy.ndarray): The true output values for each sample with shape [num_samples, 1].

        Returns:
        consequent_values (numpy.ndarray): An array of consequent values for each combination of fuzzy rule conditions
        with shape [num_combinations, 1].
        """
        num_combinations = weights.shape[1]
        numerator, denominator = np.zeros(num_combinations), np.zeros(num_combinations)
        for j in range(num_combinations):  # 0 = num samples, 1 = num combinations
            numerator[j] = np.dot(weights[:, j], output_data)
            denominator[j] = np.sum(weights[:, j])
        return (numerator / denominator).reshape(num_combinations, 1)

    def derive_linguistic_labels_and_predict(self, x, b, B, k):
        """
        Derive linguistic labels and predict output values using fuzzy inference.

        Parameters:
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
        combinations, num_combinations = self.generate_combinations(k)

        # Calculate membership values
        membership_values = self.compute_membership_values_multidim(b, B)
        membership_values_x = self.compute_membership_values_multidim(x, k)

        # Calculate degrees of compatibility
        degree_of_compatibility = np.array(self.compute_degrees_of_compatibility(membership_values_x, k))

        # Calculate the main and secondary rule tables
        main_table = np.argmax(membership_values, axis=-1)
        secondary_table = np.argsort(-membership_values, axis=-1)[:, :, 1]

        # Store the corresponding membership values
        membership_main = np.max(membership_values, axis=-1)
        membership_secondary = np.partition(membership_values, -2, axis=-1)[:, :, -2]

        # Create fuzzy sub-spaces
        space = self.generate_partitions(B[0], 0, 1)

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

    def run(self, k=np.array([5, 5]), alpha=2, B=np.array([5])):
        """
        Parameters:
        k (numpy.ndarray): Number of fuzzy sets for each feature with shape (num_features,).
        alpha (float): A positive constant that controls the power transformation.
        B (numpy.ndarray): Range of fuzzy sets for each feature with shape (num_features, num_fuzzy_sets).

        Returns:
        pi
        main_table (numpy.ndarray): Main linguistic labels for each combination of features with
            shape (num_combinations, num_features).
        secondary_table (numpy.ndarray): Secondary linguistic labels for each combination of features with
            shape (num_combinations, num_features).
        output_inferred (numpy.ndarray): Inferred output values with shape (num_samples,).
        consequent_values (numpy.ndarray): An array of consequent values for each combination of fuzzy rule conditions
        with shape [num_combinations, 1].
        """

        membership_values = self.compute_membership_values_multidim(self.input_mat, k)
        degree_of_compatibility = self.compute_degrees_of_compatibility(membership_values, k)
        weights = self.compute_weights(degree_of_compatibility, alpha)
        consequent_values = self.compute_consequent_values(weights, self.output_vec)
        output_inferred, main_table, secondary_table = self.derive_linguistic_labels_and_predict(self.input_mat,
                                                                                                 consequent_values,
                                                                                                 B,
                                                                                                 k)

        pi = (1 / self.input_mat.shape[0]) * np.sum((self.output_vec - output_inferred) ** 2)
        return pi, main_table, secondary_table, output_inferred, consequent_values

    def evaluate_performance(self, k_values, alpha_values):
        """
        Parameters:
        k_values (numpy.ndarray): List of values of k to generate k by k
        alpha_values (numpy.ndarray): Alpha values to search

        Returns:
        pi (numpy.ndarray): List of performance indices at specific alpha and k values in (number of alpha values, number of k values)
        best_pi (float):  Best performance index
        best_alpha (float):  Alpha of best performance index run
        best_k (int): K of the best performance index run
        """
        num_k_values = k_values.shape[0]
        num_alpha_values = alpha_values.shape[0]

        pi = np.zeros([num_alpha_values, num_k_values])

        for i in range(num_k_values):
            k = np.array([k_values[i], k_values[i]])
            for j in range(num_alpha_values):
                alpha = alpha_values[j]
                pi[j, i], _, _, _, _ = self.run(k, alpha, k)

        best_pi = np.min(pi)
        best_alpha = alpha_values[np.where(pi == np.min(pi))[0]][0]
        best_k = k_values[np.where(pi == np.min(pi))[1]][0]

        return pi, best_pi, best_k, best_alpha
