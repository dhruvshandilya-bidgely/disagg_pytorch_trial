"""
Author - Sahana M
Date - 2/3/2021
Initialises consumption data and vacation data required for time band detection
"""

# Import python packages

import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def init_data_matrix(in_data, seasonal_wh_config, debug):

    """
    Get (days x time) shape data matrix of consumption and vacation data
    Args:
        in_data                (np.ndarray)    : 21 column input data
        seasonal_wh_config     (dict)          : Dictionary containing all needed configuration variables
        debug                  (dict)          : Contains all variables required for debugging

    Returns:
        swh_data_matrix        (np.ndarray)    : (days x time) shape consumption data 2d array
        vacation_data_matrix   (np.ndarray)    : (days x time) shape vacation data 2d array
    """

    swh_data = deepcopy(in_data)
    init_vacation_data = deepcopy(debug['vacation_output'][:, 0:2])
    init_vacation_data = (np.logical_or(init_vacation_data[:, 0], init_vacation_data[:, 1])) * 1

    sampling_rate = seasonal_wh_config.get('user_info').get('sampling_rate')

    # get the hours in a day

    num_pd_in_day = int(Cgbdisagg.SEC_IN_DAY / sampling_rate)
    pd_mult = int(Cgbdisagg.SEC_IN_HOUR / sampling_rate)

    # Prepare day timestamp matrix and get size of all matrices

    day_ts, row_idx = np.unique(swh_data[:, Cgbdisagg.INPUT_DAY_IDX], return_inverse=True)
    day_ts = np.tile(day_ts, reps=(num_pd_in_day, 1)).transpose()

    # Initialize all 2d matrices with default value of nan except the boolean ones

    swh_data_matrix = np.full(shape=day_ts.shape, fill_value=0.0)
    vacation_data_matrix = np.full(shape=day_ts.shape, fill_value=0.0)

    # Compute hour of day based indices to use

    col_idx = swh_data[:, Cgbdisagg.INPUT_EPOCH_IDX] - swh_data[:, Cgbdisagg.INPUT_DAY_IDX]
    col_idx = col_idx / Cgbdisagg.SEC_IN_HOUR
    col_idx = (pd_mult * (col_idx - col_idx.astype(int) + swh_data[:, Cgbdisagg.INPUT_HOD_IDX])).astype(int)

    # Create day wise 2d arrays for each variable

    swh_data_matrix[row_idx, col_idx] = swh_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    vacation_data_matrix[row_idx, col_idx] = init_vacation_data

    # Keep the row_idx & col_idx for reverse mapping

    debug['row_idx'] = row_idx
    debug['col_idx'] = col_idx

    return swh_data_matrix, vacation_data_matrix
