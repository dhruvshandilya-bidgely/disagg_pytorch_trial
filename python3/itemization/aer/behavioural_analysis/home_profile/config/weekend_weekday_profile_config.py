
"""
Author - Nisha Agarwal
Date - 3rd Feb 21
Master file for calculation of occupancy profile and occupants count
"""

# Import python packages

import numpy as np


def get_weekday_weekend_profile_config(samples_per_hour=1, weekday_weekend_diff=0):

    """
       Initialize config dict containing config values for weekend profile calculation

       Parameters:
           samples_per_hour             (int)               : samples in an hour

       Returns:
           config                       (dict)              : prepared config dictionary
    """

    config = dict()

    config.update({
        "hours_bucket": [np.arange(6 * samples_per_hour, 12 * samples_per_hour + 1),
                         np.arange(11 * samples_per_hour, 13 * samples_per_hour + 1),
                         np.arange(18 * samples_per_hour, 20 * samples_per_hour + 1),
                         np.arange(10 * samples_per_hour, 18 * samples_per_hour + 1),
                         np.arange(14 * samples_per_hour, 16 * samples_per_hour + 1),
                         np.arange(7 * samples_per_hour, 9 * samples_per_hour + 1),
                         np.arange(6 * samples_per_hour, 8 * samples_per_hour + 1)],

        'late_night_hours': np.arange(1 * samples_per_hour, 5 * samples_per_hour).astype(int),
        'morn_act_hours':  np.arange(5 * samples_per_hour, 10 * samples_per_hour + 1).astype(int),

        'early_arrival_score_thres': 0.5,
        'office_goer_score_thres': 0.6,
        'home_stayer_score_thres': 0.5,
        'morn_weekend_act_score_offset': 0.1,
        'eve_weekend_act_score_offset': 0.05,
        'len_thres_for_morn_weekend_act': 0.7,
        'len_thres_for_eve_weekend_act': 0.8,
        'profile_diff_thres': [0.06, 0.1, 0.12, 0.14][np.digitize(weekday_weekend_diff, [0.2, 0.5, 0.7])],


    })


    return config
