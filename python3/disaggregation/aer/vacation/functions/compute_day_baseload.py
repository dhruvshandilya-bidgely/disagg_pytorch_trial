"""
Author - Mayank Sharan
Date - 30/8/19
Compute baseload for each day for vacation using a day wise consumption matrix
"""

# Import python packages

import copy
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.maths_utils.matlab_utils import superfast_matlab_percentile


def compute_day_baseload(day_data, vacation_config):

    """
    Computes baseload at day level with smoothing using moving average

    Parameters:
        day_data                (np.ndarray)        : 2d matrix containing day wise consumption data
        vacation_config         (dict)              : Dictionary containing all needed configuration variables

    Returns:
        day_baseload            (np.ndarray)        : Contains baseload day by day
    """

    # Extract config for baseload computation

    sampling_rate = vacation_config.get('user_info').get('sampling_rate')
    compute_baseload_config = vacation_config.get('compute_baseload')

    # Render invalid all values below the minimum valid threshold

    day_data = copy.deepcopy(day_data)
    day_data[day_data <= compute_baseload_config.get('min_value_for_baseload')] = np.nan

    # Compute day level baseload values

    daily_baseload = superfast_matlab_percentile(day_data, compute_baseload_config.get('baseload_percentile'), axis=1)

    # Compute rolling sum of daily baseload

    window_size = compute_baseload_config.get('baseload_window_size')
    baseload_cum_sum = np.nancumsum(daily_baseload)

    baseload_roll_sum = baseload_cum_sum
    baseload_roll_sum[window_size:] = baseload_cum_sum[window_size:] - baseload_cum_sum[:-window_size]

    # Average it out by number of days

    for idx in range(min(window_size - 1, len(daily_baseload))):
        daily_baseload[idx] = baseload_roll_sum[idx] / (idx + 1)

    daily_baseload[window_size - 1:] = baseload_roll_sum[window_size - 1:] / window_size

    # Make the baseload 'sampling rate agnostic' so that we can use same parameters

    daily_baseload = np.round(daily_baseload * (Cgbdisagg.SEC_IN_HOUR / sampling_rate), 2)

    return daily_baseload
