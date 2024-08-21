"""
Author - Mayank Sharan
Date - 11/1/19
This removes the non-zero minimum from data week by week
"""

# Import python packages

import copy
import numpy as np


def remove_baseload(day_data, pp_config):
    """
    Parameters:
        day_data            (np.ndarray)        : 2d matrix with each row representing a day
        pp_config           (dict)              : Dictionary containing all configuration variables for pool pump

    Returns:
        data_bl_removed     (np.ndarray)        : Day wise data matrix with baseload subtracted
    """

    baseload_window = pp_config.get('baseload_window')
    day_wise_baseload = np.full(shape=day_data.shape, fill_value=0.0)

    # Use jumping window of 7 days and get positive min to remove from all data points

    for start_idx in range(0, day_data.shape[0], baseload_window):

        end_idx = min(start_idx + baseload_window, day_data.shape[0])

        temp_data = copy.copy(day_data[start_idx: end_idx, :])
        temp_data[temp_data <= 0] = np.inf

        temp_baseload = np.min(temp_data)

        if np.isinf(temp_baseload):
            temp_baseload = 0

        day_wise_baseload[start_idx:end_idx, :] = temp_baseload

    data_bl_removed = copy.deepcopy(day_data)
    data_bl_removed -= day_wise_baseload
    data_bl_removed[data_bl_removed < 0] = 0

    return data_bl_removed
