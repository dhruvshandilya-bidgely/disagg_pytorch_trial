"""
Author - Anand Kumar Singh
Date - 19th Feb 2021
Call the HVAC inefficiency module and get output
"""

import numpy as np
from scipy.stats import mode

from python3.utils.maths_utils.find_seq import find_seq


def downsample_n_col_data(input_data, target_rate, hour_bc_columns = list()):

    """
    Parameters:
        input_data          (np.ndarray)        : N column input data, 1st column timestamp rest all col are values to be aggregated
        target_rate         (int)               : The sampling rate in seconds that the data has to be down sampled to
        hour_bc_columns     (list)               : The columns which are index and can't directly be averaged

    Returns:
        downsampled_data    (np.ndarray)        : N col array with down sampled data
    """

    # Initialize the column indices for the array to be down sampled

    ts_col_idx = 0
    val_col_idx_list = [i for i in range(1, input_data.shape[1])]

    sampling_rate = int(mode(np.diff(input_data[:, ts_col_idx]))[0][0])

    # Check a few conditions on the sampling rates to determine feasibility of down sampling

    if sampling_rate == target_rate:
        return input_data

    elif sampling_rate > target_rate or not(target_rate % sampling_rate == 0):
        print('Can\'t handle this sampling rate')
        return input_data

    else:

        group_column = (np.ceil((input_data[:, ts_col_idx] / target_rate)) * target_rate)
        new_epoch_ts, bin_idx = np.unique(group_column, return_inverse=True)

        # Calculate the last index in each unique seq to use for downsampling columns.
        # To align with end aggregated down sampling

        seq_end_col = 2

        seq_arr = find_seq(group_column, min_seq_length=0)
        new_epoch_idx = seq_arr[:, seq_end_col].astype(int)

        # Fill values in the down sampled array

        downsampled_data = np.full(shape=(len(new_epoch_ts), input_data.shape[1]), fill_value=np.nan)

        # Preparing epoch values
        downsampled_data[:, ts_col_idx] = new_epoch_ts

        for val_col_idx in val_col_idx_list:
            if val_col_idx in hour_bc_columns:
                downsampled_data[:, val_col_idx] =  input_data[new_epoch_idx, val_col_idx]
            else:
                downsampled_data[:, val_col_idx] = np.bincount(bin_idx, weights=input_data[:, val_col_idx])

        return downsampled_data


def downsample_n_col_data_average(input_data, target_rate, hour_bc_columns = []):

    """
    Parameters:
        input_data          (np.ndarray)        : N column input data, 1st column timestamp rest all col are values to be aggregated
        target_rate         (int)               : The sampling rate in seconds that the data has to be down sampled to
        hour_bc_columns     (int)               : The columns which are index and can't directly be averaged

    Returns:
        downsampled_data    (np.ndarray)        : N col array with down sampled data
    """

    # Initialize the column indices for the array to be down sampled

    ts_col_idx = 0
    val_col_idx_list = [i for i in range(1, input_data.shape[1])]

    sampling_rate = int(mode(np.diff(input_data[:, ts_col_idx]))[0][0])

    # Check a few conditions on the sampling rates to determine feasibility of down sampling

    if sampling_rate == target_rate:
        return input_data

    elif sampling_rate > target_rate or not(target_rate % sampling_rate == 0):
        print('Can\'t handle this sampling rate')
        return input_data

    else:
        group_column = (np.ceil((input_data[:, ts_col_idx] / target_rate)) * target_rate)
        new_epoch_ts, bin_idx = np.unique(group_column, return_inverse=True)

        # Calculate the last index in each unique seq to use for downsampling columns.
        # To align with end aggregated down sampling

        seq_end_col = 2

        seq_arr = find_seq(group_column, min_seq_length=0)
        new_epoch_idx = seq_arr[:, seq_end_col].astype(int)

        # Fill values in the down sampled array

        downsampled_data = np.full(shape=(len(new_epoch_ts), input_data.shape[1]), fill_value=np.nan)

        # Preparing epoch values
        downsampled_data[:, ts_col_idx] = new_epoch_ts

        for val_col_idx in val_col_idx_list:
            if val_col_idx in hour_bc_columns:
                downsampled_data[:, val_col_idx] = input_data[new_epoch_idx, val_col_idx]
            else:
                total_value = np.bincount(bin_idx, weights=input_data[:, val_col_idx])
                count_value = np.bincount(bin_idx, weights=np.ones_like(input_data[:, val_col_idx]))
                downsampled_data[:, val_col_idx] = total_value / count_value

        return downsampled_data
