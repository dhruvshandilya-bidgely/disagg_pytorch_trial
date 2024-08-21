"""
Author - Nisha Agarwal
Date - 3rd Jan 21
This file contain config values for calculating activity profile attributes
"""

# Import python packages

import numpy as np

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def init_activity_profile_config(samples_per_hours):

    """
    Initialize config used for activity profile calculation

    Parameters:
        samples_per_hour         (int)          : number of samples in an hour
    Returns:
        config                   (dict)         : active hours config dict
    """

    config = dict()

    wakeup_probable_hours = np.arange(5 * samples_per_hours, 12 * samples_per_hours + 1).astype(int)
    sleep_probable_hours = np.arange(20 * samples_per_hours, 27 * samples_per_hours + 1).astype(int) % \
                           (samples_per_hours*Cgbdisagg.HRS_IN_DAY)

    # 'activity_curve_max_perc': max percentile for calculating activity curve range,
    # 'activity_curve_min_perc': min percentile for calculating activity curve range,
    # 'sleep_probable_hours' : probable sleep hours
    # 'wakeup_probable_hours': probable wakeup hours

    general_config = {
        'activity_curve_max_perc': 97,
        'activity_curve_min_perc': 3,
        'sleep_probable_hours': sleep_probable_hours,
        'wakeup_probable_hours': wakeup_probable_hours
    }

    config.update({

        "general_config": general_config

    })

    perc_array = np.zeros(100)

    buckets = (np.array([0, 0.05, 0.17, 0.3, 0.42, 0.6, 1]) * 100).astype(int)
    perc_buckets = [65, 55, 45, 35, 25, 15, 5]

    for i in range(len(buckets) - 1):
        val = (perc_buckets[i] - perc_buckets[i + 1]) / (buckets[i + 1] - buckets[i])
        perc_array[buckets[i]: buckets[i + 1]] = np.arange(perc_buckets[i], perc_buckets[i + 1], -val)

    # 'perc_array': percentile mapping with clean days fraction,
    # 'chunk_length': length of each chunk for calculating activity curve,
    # 'vacation_limit1': max vacation days for validity of a chunk,
    # 'min_valid_chunk': minimum number of valid chunks required in the first iteration,
    # 'vacation_limit2': max vacation days for validity of a chunk in second iteration

    prepare_activity_curve_config = {
        'perc_array': perc_array,
        'chunk_length': 10,
        'vacation_limit1': 3,
        'min_valid_chunk': 3,
        'vacation_limit2': 6,
        "non_clean_user_perc": 5
    }

    config.update({
        "prepare_activity_curve_config": prepare_activity_curve_config
    })

    # list of column indexes in the activity segments array
    # 'type': type of segment,
    # segment_types = [1, 2, 3, 4, 5]
    # segment_names = ["plain", "mountain", "plateau", "uphill", "downhill"]
    # 'level': level of segment,
    # 'start': start index,
    # 'end': end index,
    # 'morning': 1 if the segment is present in morning hours,
    # 'location': time of day,
    # 'pattern': neighbouring segments,
    # 'slope': net slope of segment,
    # 'variation': diff in max and min value in segment,

    segments_config = {
        'total_keys': 9,
        'type': 0,
        'level': 1,
        'start': 2,
        'end': 3,
        'morning': 4,
        'location': 5,
        'pattern': 6,
        'slope': 7,
        'variation': 8,

    }

    config.update({
        "segments_config": segments_config
    })

    config.update({
        "default_wakeup": "7:00",
        "default_sleep": "21:00",
        "default_wakeup_int": 7,
        "default_sleep_int": 21
    })

    return config
