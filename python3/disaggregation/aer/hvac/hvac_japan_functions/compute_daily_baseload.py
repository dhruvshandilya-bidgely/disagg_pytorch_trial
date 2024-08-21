"""
Author - Mayank Sharan
Date - 08/03/2019
Compute baseload on a daily level for the given data
"""

# Import python packages

import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.maths_utils.matlab_utils import percentile_1d


def get_baseload(data):

    """
    Parameters:
        data                (np.ndarray)        : Array with values to be used to compute baseload

    Returns:
        bval                (float)             : The baseload value corresponding to the given data
    """

    min_bl = 5
    valid_data = data[data > min_bl]

    if len(valid_data) > 100:
        bval = percentile_1d(valid_data, 1)
    elif len(valid_data) > 0:
        bval = np.min(valid_data)
    else:
        bval = 0

    return bval


def compute_daily_baseload(input_data):

    """
    Parameters:
        input_data          (np.ndarray)        : 21 column input data

    Returns:
        base_load           (np.ndarray)        : 1d array with same rows as input data containing baseload values
    """

    num_points = input_data.shape[0]

    # Initialise variables to calculate baseload

    base_load = np.full(shape=(num_points,), fill_value=np.nan)
    whole_house = input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    unique_days = np.unique(input_data[:, Cgbdisagg.INPUT_DAY_IDX])

    # Calculate weekly baseload

    for idx in range(unique_days.shape[0]):
        day_idx = input_data[:, Cgbdisagg.INPUT_WEEK_IDX] == unique_days[idx]
        day_idx_and_valid = np.logical_and(day_idx, np.logical_not(np.isnan(whole_house)))

        if np.sum(day_idx_and_valid) > 0:
            whole_house_curr = whole_house[day_idx_and_valid]
            base_load[day_idx_and_valid] = get_baseload(whole_house_curr)

    # WHY DOES THIS NOT HAVE THE DS-451 FIX???

    return base_load
