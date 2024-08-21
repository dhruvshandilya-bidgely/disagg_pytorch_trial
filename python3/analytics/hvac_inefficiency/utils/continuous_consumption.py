"""
Author  -   Anand Kumar Singh
Date    -   22th Feb 2021
This file contains util functions to find continuous consumption
"""

# Import python packages

import copy
import numpy as np

# Import functions from within the project

from python3.utils.find_runs import find_runs


def get_continuous_low_consumption(consumption_array, length, min_value=0, fill_value=np.nan):

    """
        This function find continuous consumption and supresses streak of low consumption

        Parameters:
            consumption_array       (numpy.ndarray)         numpy array containing consumption value
            length                  (int)                   length of low consumption streaks
            min_value               (float)                 min value
            fill_value              (float)                 values to replace consumption streak
        Returns:
            consumption_array       (numpy.ndarray)         numpy array containing updated consumption value
    """

    # Variables initialisation for setting and resetting variables

    consumption_array_copy = copy.deepcopy(consumption_array)

    valid_idx = consumption_array_copy <= min_value

    masked_consumption = np.zeros_like(consumption_array_copy)
    masked_consumption[valid_idx] = 1

    run_values, run_starts, run_lengths = find_runs(masked_consumption.ravel())

    # Updating testing values

    valid_idx = (run_values == 1)

    run_lengths = run_lengths[valid_idx]
    run_starts = run_starts[valid_idx]

    for i in range(0, run_starts.shape[0]):
        run_length = run_lengths[i]
        if run_length >= length:
            start_index = run_starts[i]
            end_index = run_length + start_index
            consumption_array_copy[start_index:end_index] = fill_value

    return consumption_array_copy


def get_continuous_high_consumption(consumption_array, length, min_value=0, fill_value=np.nan):
    """
        This function find continuous consumption and supresses streak of low consumption

        Parameters:
            consumption_array       (numpy.ndarray)         numpy array containing consumption value
            length                  (int)                   length of low consumption streaks
            min_value               (float)                 min value
            fill_value              (float)                 values to replace consumption streak
        Returns:
            consumption_array       (numpy.ndarray)         numpy array containing updated consumption value
    """

    consumption_array_copy = np.zeros_like(consumption_array, dtype=np.float)
    consumption_array_copy[:] = fill_value

    valid_idx = consumption_array > min_value

    masked_consumption = np.zeros_like(consumption_array_copy)
    masked_consumption[valid_idx] = 1

    run_values, run_starts, run_lengths = find_runs(masked_consumption.ravel())

    # Updating testing values

    valid_idx = (run_values == 1)

    run_lengths = run_lengths[valid_idx]
    run_starts = run_starts[valid_idx]

    for i in range(0, run_starts.shape[0]):
        run_length = run_lengths[i]
        if run_length >= length:
            start_index = run_starts[i]
            end_index = run_length + start_index
            consumption_array_copy[start_index:end_index] = consumption_array[start_index:end_index]

    return consumption_array_copy
