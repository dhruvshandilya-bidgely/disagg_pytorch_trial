"""
Author - Sahana M
Date - 2/3/2021
Includes all mathematical function operations required for seasonal water heater
"""

# Import python packages

import numpy as np
from copy import deepcopy
from itertools import groupby


def median_filter(in_data, window_size):
    """
    This function is used to perform median filtering over a defined window size
    Parameters:
        in_data             (np.ndarray)           : Input data as a 2D matrix
        window_size         (int)                  : Window size to perform median filtering on
    Returns:
        filtered_data       (np.ndarray)           : Median filtered data
    """
    data = deepcopy(in_data)
    window_half = (window_size - 1) // 2
    y = np.zeros((len(data), window_size), dtype=data.dtype)
    y[:, window_half] = data

    for i in range(window_half):
        j = window_half - i
        y[j:, i] = data[: -j]
        y[:j, i] = data[0]
        y[:-j, -(i + 1)] = data[j:]
        y[-j:, -(i + 1)] = data[-1]

    filtered_data = np.median(y, axis=1)
    return filtered_data


def percentile_filter(in_data, percentile, window_size):
    """
    This function is used to perform percentile filtering over a defined window size and percentile
    Parameters:
        in_data             (np.ndarray)           : Input data as a 2D matrix
        percentile          (int)                  : Percentile value
        window_size         (int)                  : Window size to perform Percentile filtering on
    Returns:
        filtered_data       (np.ndarray)           : Percentile filtered data
    """

    data = deepcopy(in_data)
    window_half = (window_size - 1) // 2
    y = np.zeros((len(data), window_size), dtype=data.dtype)
    y[:, window_half] = data

    for i in range(window_half):
        j = window_half - i
        y[j:, i] = data[: -j]
        y[:j, i] = data[0]
        y[:-j, -(i + 1)] = data[j:]
        y[-j:, -(i + 1)] = data[-1]

    filtered_data = np.round(np.percentile(y, percentile, axis=1))

    return filtered_data


def find_seq(arr, min_seq_length=1):

    """
    Find the sequence of consecutive values
    Parameters:
        arr                 (np.ndarray)        : Array in which to find the sequences
        min_seq_length      (int)               : Set as the smallest length of the sequence to be considered a chunk
    Returns:
        res                 (np.ndarray)        : 4 column result, integer, start_idx, end_idx, num_in_seq
    """

    # Initialise the result array

    res = []
    start_idx = 0

    # Get groups

    group_list = groupby(arr)

    for seq_num, seq in group_list:

        # Get number of elements in the sequence

        seq_len = len(list(seq))

        # Discard single elements since they are not a sequence

        if seq_len < min_seq_length:
            start_idx += seq_len
            continue
        else:
            temp_res = [seq_num, start_idx, start_idx + seq_len - 1, seq_len]
            start_idx += seq_len
            res.append(temp_res)

    return np.array(res)
