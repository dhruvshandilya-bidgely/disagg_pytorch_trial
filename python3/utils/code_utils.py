"""
Author - Sahana M
Date - 09/01/2024
Code utils
"""

# Import python packages

import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def get_2d_matrix(input_data, sampling_rate):
    """
    Covert 2D timestamp data to day level data

        Parameters:
            input_data                   (np.ndarray)         : Input data
            sampling_rate                (int)                : Sampling rate
        Returns:
            day_input_data               (np.ndarray)         : day level input data
            row_idx                      (np.ndarray)         : Row indexes mapping
            col_idx                      (np.ndarray)         : Column indexes mapping
    """

    num_pd_in_day = int(Cgbdisagg.SEC_IN_DAY / sampling_rate)
    pd_mult = int(Cgbdisagg.SEC_IN_HOUR / sampling_rate)

    day_idx = Cgbdisagg.INPUT_DAY_IDX

    # Prepare day timestamp matrix and get size of all matrices

    day_ts, _, row_idx = np.unique(input_data[:, Cgbdisagg.INPUT_DAY_IDX], return_inverse=True, return_index=True)
    day_ts = np.tile(day_ts, reps=(num_pd_in_day, 1)).transpose()

    # Compute hour of day based indices to use

    col_idx = input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] - input_data[:, day_idx]
    col_idx = col_idx / Cgbdisagg.SEC_IN_HOUR
    col_idx = (pd_mult * (col_idx - col_idx.astype(int) + input_data[:, Cgbdisagg.INPUT_HOD_IDX])).astype(int)

    # Create day wise 2d arrays for each variable

    epochs_in_a_day = int(Cgbdisagg.HRS_IN_DAY * (Cgbdisagg.SEC_IN_HOUR / sampling_rate))
    no_of_days = len(day_ts)

    day_input_data = np.zeros((no_of_days, epochs_in_a_day, len(input_data[0])))

    day_input_data[row_idx, col_idx, :] = input_data

    day_input_data = np.swapaxes(day_input_data, 0, 1)
    day_input_data = np.swapaxes(day_input_data, 0, 2)

    return day_input_data, row_idx, col_idx
