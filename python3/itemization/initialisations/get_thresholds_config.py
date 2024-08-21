"""
Author - Nisha Agarwal
Date - 7th Sep 2020
Initialize general config required to maintain ts level sanity checks
"""

# Import python packages

import numpy as np


def get_thresholds_config(total_cons):

    """
    Initialize general config required to maintain ts level sanity checks

    Parameters:
        total_cons             (float)          : total monthly cons of the user
    Returns:
        config                 (dict)           : Dict containing all ts level sanity checks related params
    """

    config = dict()

    max_ts_lim = {

        'ent_arr': -1,
        'max_wh_active_pulse': 1.2,
        'max_ev_delta': 1.1,

        'max_hvac_limit': 20000,
        'wh_limit': [4000, 8000],
        'ent_ts_limit': [500, 600, 700, 800, 850, 900, 950][np.digitize(total_cons, [700, 1500, 2000, 3000, 4000, 6000])] + 200,
        'cook_ts_limit': [3500, 4000, 4500, 5000][np.digitize(total_cons, [500, 4000, 6000])],
        'ld_ts_limit': 5000,
        'perc_for_max_wh_cap': 90,
        'def_ld_ts_limit': 4000,
        'def_cook_ts_limit': 3000,
        'min_ld_amp': 400,
        'ld_amp_drop_for_missing_app': 1000,
        'cook_default_app_cons': np.array([250, 700, 100]),
        'ld_default_app_cons': np.array([1000, 1000, 1200]),
        'min_ent_cons_for_high_occ': [200, 300],
        'ent_cons_offset_based_on_occupancy': 100,
        'additional_amp_for_drier': 500,
        'min_ld_amp_for_japan': 400,
        'min_ld_amp_for_app_prof': 700,
        'min_drier_amp': 2000,
        'cook_amp_offset': 300,
        'min_cook_amp': 500,
        'min_cook_amp_for_missing_app': 400,
        'min_days_for_high_amp_wh': 150

    }

    config['max_ts_lim'] = max_ts_lim

    return config
