"""
Author - Mayank Sharan
Date - 19/1/19
Apply convolution using operator on the data
"""

# Import python packages

import copy
import numpy as np
import scipy.signal as sp

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def get_padded_signal(data):
    """Utility to get cyclic padded signal as needed for gradient"""

    rows_pad = 1
    cols_pad = 1

    n_rows, n_cols = data.shape

    data_padded = np.zeros(shape=(n_rows + 2 * rows_pad, n_cols + 2 * cols_pad))

    # Fill data as per padding
    data_padded[1: -1, 1: -1] = data

    # Minor bug here instead of shift one down we are doing shift 2 down
    data_padded[2:, 0] = data[:, -1]
    data_padded[:2, 0] = data_padded[2, 0]

    # Minor bug here instead of shift one up we are doing shift 2 up
    data_padded[: -2, -1] = data[:, 0]
    data_padded[-2:, -1] = data[-3, -1]

    # Fill first and last rows same as first and last rows of original signal
    data_padded[0, 1: -1] = data[0, :]
    data_padded[-1, 1: -1] = data[-1, :]

    return data_padded


def get_gradient(day_data, pp_config):
    """
    Parameters:
        day_data            (np.ndarray)        : Day wise data matrix
        pp_config           (dict)              : Dictionary containing all configuration variables for pool pump

    Returns:
        day_data_grad       (np.ndarray)        : Day wise data after gradient calculation
    """

    # Extract the prewitt operator and apply it on data

    prewitt_operator = pp_config.get('prewitt_operator')
    sampling_rate = pp_config.get('sampling_rate')

    num_pd_per_hr = int(Cgbdisagg.SEC_IN_HOUR / sampling_rate)

    # Extract data apply convolution and multiply with samples in hour

    day_data = copy.deepcopy(day_data)
    day_data_padded = get_padded_signal(day_data)

    day_data_grad = sp.convolve2d(day_data_padded, prewitt_operator, mode='same')
    day_data_grad = day_data_grad[1: -1, 1: -1] * num_pd_per_hr

    return day_data_grad
