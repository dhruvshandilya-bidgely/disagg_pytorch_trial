"""
Author - Mayank Sharan
Date - 30/8/19
Compute day level power for all days
"""

# Import python packages

import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def compute_day_power(day_data, day_nan_count, vacation_config):

    """
    Compute power for each day at hourly level

    Parameters:
        day_data                (np.ndarray)        : 2d matrix containing day wise consumption data
        day_nan_count           (np.ndarray)        : Array containing number of nans for each day
        vacation_config         (dict)              : Dictionary containing all needed configuration variables

    Returns:
        day_power               (np.ndarray)        : Contains power day by day
    """

    # Get total consumption and nan counts for each day

    day_wise_sum = np.nansum(day_data, axis=1)

    # Initialize and then compute power for each day

    nan_ratio_allowed = vacation_config.get('compute_power').get('max_nan_ratio_for_power')

    day_power = np.full(shape=(day_data.shape[0],), fill_value=np.nan)
    days_power_to_fill = day_nan_count <= (nan_ratio_allowed * day_data.shape[1])

    # Prepare the divisor value for all days

    day_valid_count = day_data.shape[1] - day_nan_count

    sampling_rate = vacation_config.get('user_info').get('sampling_rate')
    divisor_arr = day_valid_count[days_power_to_fill] * sampling_rate / Cgbdisagg.SEC_IN_HOUR

    # Fill power value for eligible days

    day_power[days_power_to_fill] = np.round(np.divide(day_wise_sum[days_power_to_fill], divisor_arr), 2)

    return day_power
