"""
Author - Mayank Sharan
Date - 26/11/2019
get day data converts the 21 column input data matrix into day-wise 2d matrix
"""

# Import python packages

import copy
import scipy.stats
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.maths_utils.matlab_utils import is_member


def get_day_data(input_data):

    """
    Parameters:
        input_data          (np.ndarray)        : 21 column input data matrix

    Returns:
        day_data            (np.ndarray)        : 2d matrix with each row representing a day
        month_ts            (np.ndarray)        : 2d matrix with corresponding bill cycle timestamps
        daily_ts            (np.ndarray)        : 2d matrix with corresponding day timestamps
        ts                  (np.ndarray)        : 2d matrix with corresponding epoch timestamps
    """

    # Creating a bug to mimic MATLAB here needs to be fixed with ind2sub code when pushing to release
    # Get period in seconds of the data and get the number of periods in a day

    pd = scipy.stats.mode(np.diff(input_data[:, Cgbdisagg.INPUT_EPOCH_IDX]))[0][0]
    num_pd = int(Cgbdisagg.SEC_IN_DAY / pd)

    # Create the day timestamp 2d matrix

    day_ts, day_index = np.unique(input_data[:, Cgbdisagg.INPUT_DAY_IDX], return_index=True)
    num_days = len(day_ts)

    day_ts_column = np.array([day_ts]).transpose()
    daily_ts = np.tile(day_ts_column, (1, num_pd))

    # Create the 2d matrices for month timestamp and current timestamp

    month_ts = np.tile(np.array([input_data[day_index, Cgbdisagg.INPUT_BILL_CYCLE_IDX]]).transpose(), (1, num_pd))

    ts = daily_ts + np.tile(np.array(range(num_pd)) * pd, (num_days, 1))

    # For some reason populate hod index with fractional values as per the period

    hour_idx_values = copy.deepcopy(input_data[:, Cgbdisagg.INPUT_HOD_IDX])

    for idx in range(1, len(hour_idx_values)):
        if input_data[idx - 1, Cgbdisagg.INPUT_HOD_IDX] == input_data[idx, Cgbdisagg.INPUT_HOD_IDX]:
            hour_idx_values[idx] = hour_idx_values[idx - 1] + (pd / Cgbdisagg.SEC_IN_HOUR)

    input_data[:, Cgbdisagg.INPUT_HOD_IDX] = hour_idx_values

    day_data = ts * np.nan

    # Yahan ka bug fix krenge par black box ke baad. abhi kuch beech ka

    for day_idx in range(num_days):

        data_of_day = input_data[input_data[:, Cgbdisagg.INPUT_DAY_IDX] == day_ts[day_idx], :]
        hour_data_day = np.array(range(num_pd)) * Cgbdisagg.HRS_IN_DAY / num_pd

        if_mem, idx_mem = is_member(data_of_day[:, Cgbdisagg.INPUT_HOD_IDX], hour_data_day)

        day_data_day = hour_data_day * np.nan

        sel_idx = if_mem[idx_mem >= 0]

        if data_of_day.shape[0] > len(sel_idx):
            sel_idx = np.r_[sel_idx, np.full(shape=(data_of_day.shape[0] - len(sel_idx),), fill_value=False)]

        day_data_day[idx_mem[idx_mem >= 0]] = data_of_day[sel_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX]
        day_data[day_idx, :] = day_data_day

    return day_data, month_ts, daily_ts, ts
