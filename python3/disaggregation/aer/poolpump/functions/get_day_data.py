"""
Author - Mayank Sharan
Date - 11/1/19
This converts 21 column data to day wise data matrix
"""

# Import python packages

import numpy as np
from datetime import datetime

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def get_day_data(input_data, sampling_rate):
    """
    Parameters:
        input_data          (np.ndarray)        : 21 column input data matrix
        sampling_rate       (int)               : Sampling rate of data in seconds

    Returns:
        day_data            (np.ndarray)        : 2d matrix with each row representing a day
        month_ts            (np.ndarray)        : 2d matrix with corresponding bill cycle timestamps
        daily_ts            (np.ndarray)        : 2d matrix with corresponding day timestamps
        epoch_ts            (np.ndarray)        : 2d matrix with corresponding epoch timestamps
    """

    # Here we do not take DST into consideration while reshaping the data

    num_pd_in_day = int(Cgbdisagg.SEC_IN_DAY / sampling_rate)

    # Figure out how many rows is the data going to have

    # Bug created here to make sure results match between old and new

    first_pt_date = datetime.utcfromtimestamp(input_data[0, Cgbdisagg.INPUT_EPOCH_IDX])
    first_pt_idx = int(((first_pt_date.hour + (first_pt_date.minute / 60)) * Cgbdisagg.SEC_IN_HOUR) / sampling_rate)

    num_data_pts = input_data.shape[0]

    first_day_pts = num_pd_in_day - first_pt_idx
    last_day_pts = (num_data_pts - first_day_pts) % num_pd_in_day

    # Bug handling in case we get exact days data

    if last_day_pts == 0:
        last_day_pts = num_pd_in_day

    num_rows = int(np.ceil((num_data_pts - first_day_pts) / num_pd_in_day)) + 1

    # Initialize 2d matrices

    month_ts = np.full(shape=(num_rows, num_pd_in_day), fill_value=np.nan)
    day_ts = np.full(shape=(num_rows, num_pd_in_day), fill_value=np.nan)
    epoch_ts = np.full(shape=(num_rows, num_pd_in_day), fill_value=np.nan)
    day_data = np.full(shape=(num_rows, num_pd_in_day), fill_value=0.0)

    # Fill the first row

    month_ts[0, first_pt_idx:] = input_data[: first_day_pts, Cgbdisagg.INPUT_BILL_CYCLE_IDX]
    day_ts[0, first_pt_idx:] = input_data[: first_day_pts, Cgbdisagg.INPUT_DAY_IDX]
    epoch_ts[0, first_pt_idx:] = input_data[: first_day_pts, Cgbdisagg.INPUT_EPOCH_IDX]
    if first_day_pts != num_pd_in_day:
        sampling_rate_arr = sampling_rate * np.arange(1, first_pt_idx + 1)
        fill_nan_arr = np.sort(input_data[0, Cgbdisagg.INPUT_EPOCH_IDX] - sampling_rate_arr)
        epoch_ts[0, :first_pt_idx] = fill_nan_arr
    day_data[0, first_pt_idx:] = input_data[: first_day_pts, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    # Fill the middle

    month_ts[1: -1, :] = \
        np.reshape(a=input_data[first_day_pts: num_data_pts - last_day_pts, Cgbdisagg.INPUT_BILL_CYCLE_IDX],
                   newshape=(num_rows - 2, num_pd_in_day))

    day_ts[1: -1, :] = \
        np.reshape(a=input_data[first_day_pts: num_data_pts - last_day_pts, Cgbdisagg.INPUT_DAY_IDX],
                   newshape=(num_rows - 2, num_pd_in_day))

    epoch_ts[1: -1, :] = \
        np.reshape(a=input_data[first_day_pts: num_data_pts - last_day_pts, Cgbdisagg.INPUT_EPOCH_IDX],
                   newshape=(num_rows - 2, num_pd_in_day))

    day_data[1: -1, :] = \
        np.reshape(a=input_data[first_day_pts: num_data_pts - last_day_pts, Cgbdisagg.INPUT_CONSUMPTION_IDX],
                   newshape=(num_rows - 2, num_pd_in_day))

    # Fill the last day

    month_ts[-1, :last_day_pts] = input_data[-last_day_pts:, Cgbdisagg.INPUT_BILL_CYCLE_IDX]
    day_ts[-1, :last_day_pts] = input_data[-last_day_pts:, Cgbdisagg.INPUT_DAY_IDX]
    epoch_ts[-1, :last_day_pts] = input_data[-last_day_pts:, Cgbdisagg.INPUT_EPOCH_IDX]
    if last_day_pts != num_pd_in_day:
        sampling_rate_arr = sampling_rate * np.arange(1, num_pd_in_day - last_day_pts + 1)
        fill_nan_arr = input_data[-1, Cgbdisagg.INPUT_EPOCH_IDX] + sampling_rate_arr
        epoch_ts[-1, last_day_pts:] = fill_nan_arr
    day_data[-1, :last_day_pts] = input_data[-last_day_pts:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    day_data[np.isnan(day_data)] = 0

    return month_ts, day_ts, epoch_ts, day_data
