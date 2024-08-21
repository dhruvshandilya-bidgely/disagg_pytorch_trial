"""
Author - Nikhil Singh Chauhan
Date - 1/10/19
Contains mathematical utility functions that can be used by any module
"""

import numpy as np
from copy import deepcopy


def rotating_avg(in_data, window):
    """
    Find circular moving sum of an array

    Parameters:
        in_data         (np.ndarray)        : Input data array
        window          (int)               : Sampling rate factor

    Returns:
        out             (np.ndarray)        : Moving sum of input
    """

    # Taking deepcopy of input data

    data = deepcopy(in_data)

    # Padding size for the array

    padding = window // 2

    # Pad the start and end of array with circular data

    data = np.r_[data[-padding:], data, data[:padding]]
    a = data.cumsum()
    a[window:] = a[window:] - a[:-window]

    out = a[window-1:]/window

    return out


def rotating_sum(in_data, factor):
    """
    Find circular moving sum of an array

    Parameters:
        in_data         (np.ndarray)        : Input data array
        factor          (int)               : Sampling rate factor

    Returns:
        out             (np.ndarray)        : Moving sum of input
    """

    # Taking deepcopy of input data

    data = deepcopy(in_data)

    # Window size for the sum

    window = factor + 1

    # Padding size for the array

    padding = factor // 2

    # Pad the start and end of array with circular data

    data = np.r_[data[-factor:], data, data[:factor]]
    a = data.cumsum()
    a[window:] = a[window:] - a[:-window]

    if padding == 0:
        out = a[(factor + padding):-1]
        return out

    # Subtract the padding array portion from output

    out = a[(factor + padding):-padding]

    return out


def moving_sum(in_data, window):

    """
    Parameters:
        in_data             (np.ndarray)        : Input data
        window              (int)               : Window size

    Returns:
        arr                 (np.ndarray)        : Accumulated array
    """

    data = deepcopy(in_data)

    # Take cumulative sum over the array

    arr = data.cumsum()

    # Subtract a shifted array with original array

    arr[window:] = arr[window:] - arr[:-window]

    return arr


def convolve_function(data, mask):
    """
    Parameters:
        data            (np.ndarray)    : Input data array
        mask            (np.ndarray)    : Filter to be convolved (mask)

    Returns:
        final_data      (np.ndarray)    : Convolved array
    """

    # Pad the ends with corresponding end values of filter size

    pad_size = len(mask) - 1

    # Convolve the filter

    convolved_data = np.convolve(data, mask, 'full')
    first = pad_size - pad_size // 2

    # Return the convolved array (excluding padding)

    final_data = convolved_data[first:(first + len(data))]

    return final_data


def forward_fill(arr):
    """
    This function implements pandas column-wise ffill using numpy

    Parameters:
        arr         (np.ndarray)        :       input 2d array

    Return:
        arr         (np.ndarray)        :       2d array with nan values filled
    """

    # Here we are trying to fill nan values with the non-nan value in previous row and same column

    mask = np.isnan(arr)

    # Working of the code written below:
    # First, we create a 2-d array of the same shape as original array. Every column in this array
    # will be np.arange(0, mask.shape[0]). We then change nan values to zero and apply np.maximum accumulate
    # column-wise, which will replace 0 values with the index of nearest non-zero value before the current index.

    updated_indices = np.where(~mask, np.tile(np.arange(mask.shape[0]).reshape(-1, 1), mask.shape[1]), 0)
    np.maximum.accumulate(updated_indices, axis=0, out=updated_indices)

    # Replacing value in nan indices with values in updated indices calculated above
    arr[mask] = arr[updated_indices[mask], np.nonzero(mask)[1]]

    return arr


def create_pivot_table(data, index, columns, values):
    """
    This function implements pandas pivot table using numpy
    disclaimer: doesn't work when there are duplicate index and column values

    Parameters:
        data                (np.ndarray)    :       input 2d array
        index               (int)           :       index of the column which will act as row of pivot table
        columns             (int)           :       index of the column which will act as column of pivot table
        values              (int)           :       index of the column which will act as filled values in pivot table

    Return:
        pivot_table         (np.ndarray)    :       2d array containing the pivot table
        row                 (np.ndarray)    :       Array of day rows
        col                 (np.ndarray)    :       Array of day columns
    """

    # TODO: (Paras) Update this function to work for all sampling rates

    row, row_idx_pos = np.unique(data[:, index], return_inverse=True)
    col, col_idx_pos = np.unique(data[:, columns], return_inverse=True)

    pivot_table = np.full((len(row), len(col)), fill_value=np.nan, dtype=np.float)
    pivot_table[row_idx_pos, col_idx_pos] = data[:, values]
    return pivot_table, row, col


def merge_np_arrays_bycol(arr1, arr2, index1, index2):
    """
    This function merges two numpy arrays based on a column

    Parameters:
        arr1                    (np.ndarray)       : 2-d numpy array to be merged
        arr2                    (np.ndarray)       : 2-d numpy array to be merged
        index1                  (int)              : index of the arr1 column used for merging array
        index2                  (int)              : index of the arr2 column used for merging array

    Return:
        merged_arr              (np.ndarray)       : Merged array
    """

    # Mask of matches in arr1 against arr2
    arr1mask = np.isin(arr1[:, index1], arr2[:, index2])

    # Mask of matches in arr2 against arr1
    arr2mask = np.isin(arr2[:, index2], arr1[:, index1])

    # Mask respective arrays and concatenate for final output
    merged_arr = np.c_[arr1[arr1mask], arr2[arr2mask, 1:]]

    return merged_arr
