"""
Author - Nisha Agarwal
Date - 3rd Jan 21
This file contain config values for calculating activity levels
"""

# No imports


def init_activity_levels_config():

    """
    Initialize config used for active hours calculation

    Returns:
        config                   (dict)         : active hours config dict
    """

    config = dict()

    # 'threshold_array': parameters used for calculation of cluster center distance using activity curve range
    # 'activity_curve_diff_arr':
    # 'min_range': minimum activity curve range
    # 'increament': Parameters used for updating cluster center distances for morning segments, and higher sampling rate data
    # 'decreament':

    levels_config = {
        'threshold_array': [0.015, 0.017, 0.023, 0.025, 0.03, 0.03, 0.03, 0.035, 0.035, 0.04, 0.05, 0.05, 0.05],
        'activity_curve_diff_arr': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 1],
        'min_range': 0.01,
        'increament': 0.01,
        'decreament': 0.01
    }

    config["levels_config"] = levels_config

    return config
