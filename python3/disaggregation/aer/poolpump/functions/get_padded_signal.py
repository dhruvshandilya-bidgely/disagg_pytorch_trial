"""
Author - Mayank Sharan
Date - 11/1/19
Apply circular padding to the input signal
"""

# Import python packages

import copy
import numpy as np


def get_padded_signal(data, n_rows_pad=1, n_cols_pad=1):
    """
    Parameters:
        data                (np.ndarray)        : Matrix on which padding is to be applied
        n_rows_pad          (int)               : Number of rows tp be padded
        n_cols_pad          (int)               : number of columns to be padded

    Returns:
        data_padded         (np.ndarray)        : Matrix with padded data
    """

    # Initialize variables for data padding

    data = copy.deepcopy(data)

    num_rows, num_cols = data.shape

    num_pad_rows = num_rows + 2 * n_rows_pad
    num_pad_cols = num_cols + 2 * n_cols_pad

    data_padded = np.zeros(shape=(num_pad_rows, num_pad_cols))

    # Start adding padded data

    data_padded[n_rows_pad: num_rows + n_rows_pad, n_cols_pad: num_cols + n_cols_pad] = data

    # Rows padded in start and end are the first row and last row itself respectively

    data_padded[: n_rows_pad + 1, n_cols_pad: num_cols + n_cols_pad] = data[0, :]
    data_padded[num_rows + n_rows_pad:, n_cols_pad: num_cols + n_cols_pad] = data[-1, :]

    # Column padded in start and end are the first few columns and last few columns itself respectively

    data_padded[n_rows_pad + 1: n_rows_pad + num_rows, :n_cols_pad] = data[:-1, num_cols - n_cols_pad: num_cols]
    data_padded[n_rows_pad: n_rows_pad + num_rows - 1, num_cols + n_cols_pad:] = data[1:num_rows, :n_cols_pad]

    # Fill out the corners Top Left, Top Right, Bottom Left then Bottom right

    data_padded[:n_rows_pad + 1, : n_cols_pad] = \
        np.reshape(data_padded[:n_rows_pad + 1, n_cols_pad], newshape=(n_rows_pad + 1, 1))

    data_padded[:n_rows_pad, num_pad_cols - n_cols_pad:] = \
        np.reshape(data_padded[:n_rows_pad, num_pad_cols - (n_cols_pad + 1)], newshape=(n_rows_pad, 1))

    data_padded[num_pad_rows - n_rows_pad:, :n_cols_pad] = \
        np.reshape(data_padded[num_pad_rows - n_rows_pad:, n_cols_pad], newshape=(n_rows_pad, 1))

    data_padded[num_pad_rows - (n_rows_pad + 1):, num_pad_cols - n_cols_pad:] = \
        np.reshape(data_padded[num_pad_rows - (n_rows_pad + 1):, num_pad_cols - (n_cols_pad + 1)],
                   newshape=(n_rows_pad + 1, 1))

    return data_padded
