"""
Author - Sahana M
Date - 3/3/2021
Returns the time zones which satisfy min correlation threshold based on local maxima's
"""

# Import python packages

import numpy as np
from copy import deepcopy


def get_time_zones(moving_band_correlation, correlation_thr, seasonal_wh_config):
    """
    Returns time zones which have local maxima satisfying min correlation threshold
    Args:
        moving_band_correlation     (np.array)      : Contains the correlation of all 12 bands
        correlation_thr             (float)         : Correlation threshold to select local maxima's
        seasonal_wh_config          (dict)          : Dictionary containing all needed configuration variables

    Returns:
        time_zones                  (np.array)      : Contains the start and end indexes of all the selected time zones
        local_max_idx               (np.array)      : Contains the indexes of all the local max indexes
    """

    moving_band_corr = deepcopy(moving_band_correlation)
    combine_hours = seasonal_wh_config.get('config').get('combine_hours')

    # Get local maxima's

    local_max_min = np.diff(np.sign(np.diff(np.r_[0, moving_band_corr, 0])))

    # Make all the values below the correlation threshold -5 for notation purpose

    local_max_min[np.where(np.array(moving_band_corr) < correlation_thr)] = -5

    # Get all the local max indexes (denoted by values -1, -2)

    local_max_idx = np.where(np.isin(local_max_min, [-1, -2]))

    # Initialise an empty time zones array

    time_zones = []

    # ------------------------------------------ STAGE 1: IDENTIFY THE TIME ZONES -------------------------------------

    for idx in local_max_idx[0]:

        left_idx = idx - 1

        # Move left until you find another local max index

        while (left_idx >= 0) and (local_max_min[left_idx] != -5) and (local_max_min[left_idx] != -2):
            left_idx -= 1

        right_idx = idx + 1

        # Move right until you find another local max index

        while (right_idx < len(local_max_min)) and (local_max_min[right_idx] != -5) and (local_max_min[right_idx] != -2):
            right_idx += 1

        # Get the range of this zone

        zone = [left_idx + 1, right_idx - 1]

        # Store this zone in the final time_zones array

        time_zones.append(zone)

    # Make sure that the time_zones are unique

    if len(time_zones) > 1:
        time_zones = np.unique(time_zones, axis=0)

    # ------------------------------------------ STAGE 2: COMBINE THE TIME ZONES --------------------------------------

    try:

        # Initialise a final_time_zone array

        final_time_zones = []

        index = 0

        while index < len(time_zones) and len(time_zones) > 1:

            j = index + 1

            # If the any 2 time bands cover a time of 4 hours or less then combine them

            while j < len(time_zones) and time_zones[j][1] - time_zones[index][0] <= combine_hours:
                j += 1

            # Get the new combined time zone indexes

            combined_zone = [time_zones[index][0], time_zones[j - 1][1]]

            # Append it to the final time zones array

            final_time_zones.append(combined_zone)

            index = j

        if len(time_zones) == 1:
            final_time_zones = time_zones

        return final_time_zones, local_max_idx

    except:

        return time_zones, local_max_idx
