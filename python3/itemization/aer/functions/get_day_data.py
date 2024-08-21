
"""
Author - Nisha Agarwal/Mayank Sharan
Date - 7th Sep 2020
Convert 1 D consumption array to 2-D day level data
"""

# Import python packages

import scipy
import numpy as np

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def get_hybrid_day_data(input_data, output_data, sampling_rate):

    """
    Covert 2 D timestamp data to day level data

        Parameters:
            input_data                   (np.ndarray)         : timestamp level input data
            output_data                  (np.ndarray)         : timestamp level disagg output data
            sampling_rate                (int)                : sampling rate of the user

        Returns:
            day_input_data               (np.ndarray)         : day level input data
            day_output_data              (np.ndarray)         : day level disagg output data
    """

    pd = scipy.stats.mode(np.diff(input_data[:, Cgbdisagg.INPUT_EPOCH_IDX]))[0][0]
    num_pd = int(Cgbdisagg.SEC_IN_DAY / pd)

    num_pd_in_day = int(Cgbdisagg.SEC_IN_DAY / sampling_rate)
    pd_mult = int(Cgbdisagg.SEC_IN_HOUR / sampling_rate)

    day_idx = Cgbdisagg.INPUT_DAY_IDX

    # Prepare day timestamp matrix and get size of all matrices

    day_ts, row_idx2, row_idx = np.unique(input_data[:, Cgbdisagg.INPUT_DAY_IDX], return_inverse=True, return_index=True)
    day_ts = np.tile(day_ts, reps=(num_pd_in_day, 1)).transpose()

    month_ts = np.tile(np.array([input_data[row_idx2, Cgbdisagg.INPUT_BILL_CYCLE_IDX]]).transpose(), (1, num_pd))

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

    day_output_data = np.zeros((no_of_days, epochs_in_a_day, len(output_data[0])))

    day_output_data[row_idx, col_idx, :] = output_data

    day_output_data = np.swapaxes(day_output_data, 0, 1)
    day_output_data = np.swapaxes(day_output_data, 0, 2)

    return day_input_data, day_output_data, month_ts



def get_day_data(input_data, sampling_rate):

    """
    Covert 2 D timestamp data to day level data

        Parameters:
            input_data                   (np.ndarray)         : timestamp level input data
            output_data                  (np.ndarray)         : timestamp level disagg output data
            sampling_rate                (int)                : sampling rate of the user

        Returns:
            day_input_data               (np.ndarray)         : day level input data
            day_output_data              (np.ndarray)         : day level disagg output data
    """

    num_pd_in_day = int(Cgbdisagg.SEC_IN_DAY / sampling_rate)
    pd_mult = int(Cgbdisagg.SEC_IN_HOUR / sampling_rate)

    day_idx = Cgbdisagg.INPUT_DAY_IDX

    # Prepare day timestamp matrix and get size of all matrices

    day_ts, row_idx2, row_idx = np.unique(input_data[:, Cgbdisagg.INPUT_DAY_IDX], return_inverse=True, return_index=True)
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

    return day_input_data
