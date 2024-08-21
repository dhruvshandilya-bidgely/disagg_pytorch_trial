"""
Author: Mayank Sharan
Created: 10-Feb-2020
Has mathematical utility functions to be used
"""

# Import python packages

import numpy as np
from itertools import groupby


def percentile_of_arr(percentile_array, value):

    """Utility to get percentile that a value lies in a range"""

    less_bool = percentile_array <= value
    more_bool = percentile_array >= value

    if value < percentile_array[0]:
        value_percentile = 0.01

    elif value > percentile_array[-1]:
        value_percentile = 99.99

    else:
        last_less_idx = np.where(less_bool)[0][-1]
        first_more_idx = np.where(more_bool)[0][0]

        lower_p = last_less_idx * 10
        upper_p = first_more_idx * 10

        cons_range = percentile_array[first_more_idx] - percentile_array[last_less_idx]
        lower_diff = value - percentile_array[last_less_idx]
        upper_diff = percentile_array[first_more_idx] - value

        value_percentile = (lower_p * upper_diff / cons_range) + (upper_p * lower_diff / cons_range)

    return value_percentile


def convolve_function(vector_1, vector_2):

    """Utility to convolve 2 1d vectors"""

    n_pad = len(vector_2) - 1

    full = np.convolve(vector_1, vector_2, 'full')

    first = n_pad - (n_pad // 2)

    return full[first:(first + len(vector_1))]


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


def rolling_sum(data, window):

    """
    Utility function to calculate rolling sum
    Parameters:
        data                (np.ndarray)        : Input data
        window              (int)               : Window size
    Returns:
        arr                 (np.ndarray)        : Accumulated array
    """

    # Take cumulative sum over the array

    arr = data.cumsum()

    # Subtract a shifted array with original array

    arr[window:] = arr[window:] - arr[:-window]

    return arr


def percentile_1d(arr, ptile):

    """
    1d percentile based on MATLAB implemented for python
    Parameters:
        arr             (np.ndarray)        : Array to calculate percentile on
        ptile           (float)             : Percentile value to calculate for
    Returns:
        val             (float)             : Percentile value
    """

    arr_clean = np.sort(arr[~np.isnan(arr)])
    num_el = len(arr_clean)

    if num_el > 0:
        p_rank = 100.0 * (np.arange(num_el) + 0.5) / num_el
        return np.interp(ptile, p_rank, arr_clean)
    else:
        return np.nan


def nan_percentile(data, ptile, axis=0):

    """
    Super fast vectorized implementation in python of MATLAB's percentile function
    Parameters:
        data            (np.ndarray)        : Array to calculate percentile on
        ptile           (float)             : Percentile value to calculate for
        axis            (int)               : Axis along which to calculate percentile
    Returns:
        val             (float)             : Percentile value
    """

    if data.ndim == 1:

        return percentile_1d(data, ptile)

    elif data.ndim == 2:

        if data.shape[axis] == 0:
            num_out = data.shape[1 - axis]
            res = np.zeros(shape=(num_out,))
            res[:] = np.nan
            return res

        quantile = float(ptile) / 100

        valid_obs = np.sum(np.isfinite(data), axis=axis)
        data_s = np.sort(data, axis=axis)

        k_arr = quantile * valid_obs - 0.5
        k_arr[k_arr < 0] = 0
        k_arr[k_arr > valid_obs - 1] = valid_obs[k_arr > valid_obs - 1] - 1

        f_arr = np.floor(k_arr).astype(np.int32)
        c_arr = np.ceil(k_arr).astype(np.int32)
        fc_equal_k_mask = f_arr == c_arr

        num_out = data.shape[1 - axis]
        res = np.zeros(shape=(num_out, ))

        if axis == 1:
            floor_val = data_s[np.arange(num_out), f_arr] * (c_arr - k_arr)
            ceil_val = data_s[np.arange(num_out), c_arr] * (k_arr - f_arr)
            res = floor_val + ceil_val
            res[fc_equal_k_mask] = (data_s[np.arange(num_out), f_arr])[fc_equal_k_mask]

        if axis == 0:
            floor_val = data_s[f_arr, np.arange(num_out)] * (c_arr - k_arr)
            ceil_val = data_s[c_arr, np.arange(num_out)] * (k_arr - f_arr)
            res = floor_val + ceil_val
            res[fc_equal_k_mask] = (data_s[f_arr, np.arange(num_out)])[fc_equal_k_mask]

    elif data.ndim == 3 and axis == 2:

        quantile = float(ptile) / 100

        valid_obs = np.sum(np.isfinite(data), axis=axis)
        data_s = np.sort(data, axis=axis)

        k_arr = quantile * valid_obs - 0.5
        k_arr[k_arr < 0] = 0
        k_arr[k_arr > valid_obs - 1] = valid_obs[k_arr > valid_obs - 1] - 1

        f_arr = np.floor(k_arr).astype(np.int32)
        c_arr = np.ceil(k_arr).astype(np.int32)
        fc_equal_k_mask = f_arr == c_arr

        num_out_x = data.shape[0]
        num_out_y = data.shape[1]

        floor_val = data_s[np.tile(np.arange(num_out_x), (num_out_y, 1)).transpose(),
                           np.tile(np.arange(num_out_y), (num_out_x, 1)), f_arr] * (c_arr - k_arr)
        ceil_val = data_s[np.tile(np.arange(num_out_x), (num_out_y, 1)).transpose(),
                          np.tile(np.arange(num_out_y), (num_out_x, 1)), c_arr] * (k_arr - f_arr)

        res = floor_val + ceil_val
        res[fc_equal_k_mask] = (data_s[np.tile(np.arange(num_out_x), (num_out_y, 1)).transpose(),
                                       np.tile(np.arange(num_out_y), (num_out_x, 1)), f_arr])[fc_equal_k_mask]
    else:
        res = np.array([])

    return res
