"""
Author - Mayank Sharan
Date - 31/1/19
Get season mapping for each day in the data
"""

# Import python packages

import copy
import numpy as np

# Import packages from the project

from python3.disaggregation.aer.poolpump.functions.cleaning_utils import find_edges


def get_seasonal_segments(input_data, pp_config):
    """
    Parameters:
        input_data          (np.ndarray)        : 21 column input data
        pp_config           (dict)              : Dictionary containing all configuration variables for pool pump

    Returns:
        day_seasons         (np.ndarray)        : Array containing mapping of season for each day
    """

    # Extract constants form config

    set_point = 65
    cdd = 74
    hdd = 56

    input_data = copy.deepcopy(input_data)

    # Consider only non-NAN temperatures
    temp_data = copy.deepcopy(input_data)
    temp_data = temp_data[~np.isnan(temp_data[:, 7])]

    unq_months, months_idx, months_count = np.unique(temp_data[:, 0], return_counts=True, return_inverse=True)

    avg_temp = np.bincount(months_idx, temp_data[:, 7]) / months_count

    monthly_avg_temp = np.hstack((unq_months.reshape(-1, 1), avg_temp.reshape(-1, 1), np.zeros((len(avg_temp), 1))))

    monthly_avg_temp = monthly_avg_temp[np.lexsort((monthly_avg_temp[:, 0], monthly_avg_temp[:, 1]))]

    temp_diff = np.abs(monthly_avg_temp[:, 1] - set_point)
    transition_limit = sorted(temp_diff)[int(4 * len(unq_months) / 12) - 1]
    trans_idx = np.where(temp_diff <= transition_limit)[0]

    wtr_cutoff = min(monthly_avg_temp[trans_idx[0], 1], hdd)
    smr_cutoff = max(monthly_avg_temp[trans_idx[-1], 1], cdd)

    # Below 30th percentile is Winter and Above 70th percentile is Summer

    # Assigning Season Tag
    monthly_avg_temp[monthly_avg_temp[:, 1] < wtr_cutoff, 2] = 1
    monthly_avg_temp[(monthly_avg_temp[:, 1] >= wtr_cutoff) & (monthly_avg_temp[:, 1] <= smr_cutoff), 2] = 2
    monthly_avg_temp[monthly_avg_temp[:, 1] > smr_cutoff, 2] = 3

    unique_days, days_idx = np.unique(input_data[:, 2], return_index=True)
    day_month_ts = input_data[days_idx, 0]

    num_days = len(unique_days)

    day_seasons = np.c_[
        day_month_ts, np.arange(1, num_days + 1).reshape(-1, 1), np.zeros(shape=(num_days, 1))]

    monthly_avg_temp = monthly_avg_temp[monthly_avg_temp[:, 0].argsort()]

    for i in range(monthly_avg_temp.shape[0]):
        month = monthly_avg_temp[i, 0]
        season = monthly_avg_temp[i, 2]

        day_seasons[day_seasons[:, 0] == month, 2] = season

    day_seasons_copy = day_seasons[:, 2].copy()
    day_seasons_copy[day_seasons_copy > 0] = -1
    start_arr, end_arr = find_edges(day_seasons_copy)

    for idx in range(len(start_arr)):
        day_seasons[start_arr[idx]:end_arr[idx], 2] = 2
        if (start_arr[idx] != 0) and (end_arr[idx] != len(day_seasons)) and (
                day_seasons[start_arr[idx] - 1, 2] == day_seasons[end_arr[idx] + 1, 2]):
            day_seasons[start_arr[idx]:end_arr[idx], 2] = day_seasons[start_arr[idx] - 1, 2]

    return np.array(day_seasons[:, 2], dtype=int)
