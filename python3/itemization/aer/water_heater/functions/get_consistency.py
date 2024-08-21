"""
Author - Sahana M
Date - 3/3/2021
Returns the time zones which satisfy min correlation threshold based on local maxima's
"""

# Import python packages

import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.utils.maths_utils.maths_utils import moving_sum


def get_consistency(in_data, start_time, end_time, debug, seasonal_wh_config):

    """
    Returns 1d array of energy consumption start consistency
    Args:
        in_data                     (np.ndarray)  : Contains (days x time zone) shape consumption matrix
        start_time                  (int)         : Starting index of the time zone
        end_time                    (int)         : Ending index of the time zone
        debug                       (dict)        : Contains all variables required for debugging
        seasonal_wh_config          (dict)        : Dictionary containing all needed configuration variables

    Returns:
        hourly_consistency          (np.array)    : Contains consumption consistency at hourly level
        start_hod_count             (np.array)    : Contains consumption consistency at time division level
        max_median                  (float)       : The difference between the max and median values of start_hod_count

    """

    # Initialise the required variables

    data = deepcopy(in_data)
    factor = debug.get('factor')
    wh_present_idx = deepcopy(debug.get('wh_present_idx'))
    min_amplitude = deepcopy(seasonal_wh_config.get('config').get('min_amplitude'))

    window_size = factor + 1

    # get the data of interest (only wh present days)

    consistence_check_data = data[wh_present_idx, start_time: end_time]

    # convert consumption data to 1 d array

    band_data_1d = consistence_check_data.ravel().reshape(-1, 1)

    # create a time division array

    time_div = np.tile(np.arange(0, consistence_check_data.shape[1], 1), consistence_check_data.shape[0])

    band_data_1d = np.c_[band_data_1d, time_div]

    # Mark all the indexes which satisfy a minimum amplitude as valid

    box_energy_idx = (band_data_1d[:, 0] > (min_amplitude / factor))

    # Identify those indexes

    box_energy_idx_diff = np.diff(np.r_[0, box_energy_idx.astype(int), 0])
    box_start_idx = (box_energy_idx_diff[:-1] > 0)

    # Get the consistency

    edges = np.arange(0, consistence_check_data.shape[1] + 2) - 0.5
    start_hod_count, _ = np.histogram(band_data_1d[box_start_idx, 1], bins=edges)
    start_hod_count = start_hod_count / consistence_check_data.shape[0]

    # Get hourly consistency

    hourly_consistency = moving_sum(start_hod_count, window_size)
    hourly_consistency = min(np.nanmax(hourly_consistency), 1)

    # Calculate the Max, Median difference to get the values of dispersion

    max_median = np.nanmax(start_hod_count) - np.nanmedian(start_hod_count)

    return hourly_consistency, start_hod_count, max_median
